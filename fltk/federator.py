import datetime
from distutils.command.config import config
from math import dist
import time
from typing import List

from dataclass_csv import DataclassWriter
from torch.distributed import rpc

from fltk.client import Client
from fltk.datasets.data_distribution import distribute_batches_equally
from fltk.strategy.client_selection import random_selection
from fltk.util.arguments import Arguments
from fltk.util.base_config import BareConfig
from fltk.util.data_loader_utils import load_train_data_loader, load_test_data_loader, \
    generate_data_loaders_from_distributed_dataset
from fltk.util.fed_avg import fed_average_nn_parameters
from fltk.util.log import FLLogger
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging

from fltk.util.results import EpochData
from fltk.util.tensor_converter import convert_distributed_data_into_numpy
from fltk.util.update_dist import update_dist, cal_dist_entropy

logging.basicConfig(level=logging.DEBUG)

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)

class ClientRef:
    ref = None
    name = ""
    data_size = 0
    tb_writer = None

    def __init__(self, name, ref, tensorboard_writer):
        self.name = name
        self.ref = ref
        self.tb_writer = tensorboard_writer

    def __repr__(self):
        return self.name

class Federator:
    """
    Central component of the Federated Learning System: The Federator

    The Federator is in charge of the following tasks:
    - Have a copy of the global model
    - Client selection
    - Aggregating the client model weights/gradients
    - Saving all the metrics
        - Use tensorboard to report metrics
    - Keep track of timing

    """
    clients: List[ClientRef] = []
    epoch_counter = 0
    client_data = {}
    previous_weights = {}
    comm_round = 0
    entropies = []              # Entropy of the distribution of configurations
    max_grads = []              # Maximum gradients of the distribution in every round
    dist_lr_type = "aggressive" # Three types: "constant", "adaptive" and "aggressive"

    def __init__(self, client_id_triple, num_epochs = 3, config=None):
        log_rref = rpc.RRef(FLLogger())
        self.log_rref = log_rref
        self.num_epoch = num_epochs
        self.config = config
        self.tb_path = config.output_location
        self.ensure_path_exists(self.tb_path)
        self.tb_writer = SummaryWriter(f'{self.tb_path}/{config.experiment_prefix}_federator')
        self.create_clients(client_id_triple)
        self.config.init_logger(logging)

    def create_clients(self, client_id_triple):
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref, rank=rank, world_size=world_size, config=self.config))
            writer = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer))
            self.client_data[id] = []

    def select_clients(self, n = 2):
        return random_selection(self.clients, n)

    def ping_all(self):
        for client in self.clients:
            logging.info(f'Sending ping to {client}')
            t_start = time.time()
            answer = _remote_method(Client.ping, client.ref)
            t_end = time.time()
            duration = (t_end - t_start)*1000
            logging.info(f'Ping to {client} is {duration:.3}ms')

    def rpc_test_all(self):
        for client in self.clients:
            res = _remote_method_async(Client.rpc_test, client.ref)
            while not res.done():
                pass

    def client_load_data(self):
        for client in self.clients:
            _remote_method_async(Client.init_dataloader, client.ref)

    def clients_ready(self):
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            for client in self.clients:
                if client.name not in ready_clients:
                    responses.append((client, _remote_method_async(Client.is_ready, client.ref)))
            all_ready = True
            for res in responses:
                result = res[1].wait()
                if result:
                    logging.info(f'{res[0]} is ready')
                    ready_clients.append(res[0])
                else:
                    logging.info(f'Waiting for {res[0]}')
                    all_ready = False

            time.sleep(2)
        logging.info('All clients are ready')

    def remote_run_epoch(self, epochs):
        responses = []
        client_weights = []
        chosen_configs = []
        losses = []
        test_datasizes = []
        train_datasizes = []
        selected_clients = self.select_clients(self.config.clients_per_round)
        for client in selected_clients:
            responses.append((client, _remote_method_async(Client.run_epochs, client.ref, num_epoch=epochs)))
        self.epoch_counter += epochs
        self.comm_round += 1
        for res in responses:
            epoch_data, weights = res[1].wait()

            # Receive index of chosen configuration in distribution.
            numberchosen = epoch_data.chosen_config_index
            chosenconfig = self.config.configs[numberchosen]
            batch_size = chosenconfig[0]
            lr = chosenconfig[1]
            momentum = chosenconfig[2]
            dropouts = chosenconfig[3]
            chosen_configs.append(chosenconfig)
            
            losses.append(epoch_data.loss)
            test_datasizes.append(epoch_data.test_datasize)
            train_datasizes.append(epoch_data.batch_size)
            self.client_data[epoch_data.client_id].append(epoch_data)
            # logging.info(f'{res[0]} had a batch size of {batch_size}')
            # logging.info(f'{res[0]} had a learning rate of {lr}')
            # logging.info(f'{res[0]} had a test data size of {epoch_data.test_datasize}')
            # logging.info(f'{res[0]} had a loss of {epoch_data.loss}')
            # logging.info(f'{res[0]} had a epoch data of {epoch_data}')

            res[0].tb_writer.add_scalar('training loss',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            res[0].tb_writer.add_scalar('accuracy',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            client_weights.append(weights)
        
        # Calculate the entropy of the current distribution
        self.config.old_entropy = cal_dist_entropy(self.config.dist)

        # Aggregate weights
        updated_weights = fed_average_nn_parameters(self.previous_weights, client_weights, train_datasizes, self.config.server_lr)

        # Store the weights for the next round
        self.previous_weights = updated_weights

        # Update server learning rate
        self.config.server_lr = pow(self.config.server_gamma, self.comm_round)

        # Update distributions
        self.config.dist, self.max_grads = update_dist(self.config.dist, self.config.configs, chosen_configs, losses, test_datasizes, self.max_grads, self.dist_lr_type)
        # print(f"Updated distribution: {self.config.dist}")

        # Calculate the entropy of the updated distribution
        self.config.entropy = cal_dist_entropy(self.config.dist)
        self.entropies.append(self.config.entropy)
        print(f"Entropies: {self.entropies}")

        # Send weights to the clients
        responses = []
        for client in self.clients:
            responses.append(
                (client, _remote_method_async(Client.update_nn_parameters, client.ref, new_params=updated_weights)))

        for res in responses:
            res[1].wait()
        # logging.info('Weights are updated')

        # Send distribution to the clients
        responses = []
        for client in self.clients:
            responses.append(
                (client, _remote_method_async(Client.update_client_dist, client.ref, new_dist=self.config.dist)))

        for res in responses:
            res[1].wait()
        # logging.info('Distribution is updated')

    def update_client_data_sizes(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.get_client_datasize, client.ref)))
        for res in responses:
            res[0].data_size = res[1].wait()
            logging.info(f'{res[0]} had a result of datasize={res[0].data_size}')

    def remote_test_sync(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.test, client.ref)))

        for res in responses:
            accuracy, loss, class_precision, class_recall = res[1].wait()
            logging.info(f'{res[0]} had a result of accuracy={accuracy}')

    def save_epoch_data(self):
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)
        for key in self.client_data:
            filename = f'{file_output}/{key}_epochs.csv'
            logging.info(f'Saving data at {filename}')
            with open(filename, "w") as f:
                w = DataclassWriter(f, self.client_data[key], EpochData)
                w.write()

    def ensure_path_exists(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def save_entropies(self):
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)
        filename= f'{file_output}/{len(self.clients)}c_{self.epoch_counter}e_{self.dist_lr_type}_entropies.csv'
        logging.info(f'Saving data at {filename}')
        with open(filename, "w") as f:
            for entropy in self.entropies:
                f.write(f'{entropy}\n')

    def save_time(self, total_time):
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)
        filename= f'{file_output}/{len(self.clients)}c_{self.epoch_counter}e_{self.dist_lr_type}_time.csv'
        logging.info(f'Saving data at {filename}')
        with open(filename, "w") as f:
                f.write(f'{total_time}\n')
                
    def run(self):
        """
        Main loop of the Federator
        :return:
        """
        # # Make sure the clients have loaded all the data
        self.client_load_data()
        self.ping_all()
        self.clients_ready()
        self.update_client_data_sizes()

        epoch_to_run = self.num_epoch
        addition = 0
        epoch_to_run = self.config.epochs
        epoch_size = self.config.epochs_per_cycle
        start = time.time()
        for epoch in range(epoch_to_run):
            # If the distribution entropy > threshold, execute remote_run_epoch 
            if (self.config.entropy > self.config.entropy_threshold):
                print(f'Running epoch {epoch}')
                self.remote_run_epoch(epoch_size)
                addition += 1
            else:
                break
        end = time.time()
        total_time = (end - start)*1000
        logging.info('Saving total time')
        self.save_time(total_time)

        logging.info('Printing client data')
        print(self.client_data)

        logging.info(f'Saving data')
        self.save_epoch_data()
        self.save_entropies()
        logging.info(f'Federator is stopping')

