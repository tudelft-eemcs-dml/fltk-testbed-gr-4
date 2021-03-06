import torch.nn.functional as F
import torch
import json
from fltk.util.update_dist import cal_dist_entropy

# Setting the seed for Torch
import yaml

from fltk.nets import Cifar10CNN, FashionMNISTCNN, Cifar100ResNet, FashionMNISTResNet, Cifar10ResNet, Cifar100VGG
from fltk.util.choose_config import setup_configs

SEED = 1
torch.manual_seed(SEED)

class Arguments:

    def __init__(self, logger):
        self.logger = logger

        self.batch_size = 10
        self.test_batch_size = 1000
        self.epochs = 1
        self.lr = 0.001
        self.momentum = 0.9
        self.cuda = False
        self.shuffle = True
        self.log_interval = 10
        self.kwargs = {}
        self.contribution_measurement_round = 1
        self.contribution_measurement_metric = 'Influence'

        # Hyperparameters we are tuning
        self.batch_sizes = [10, 16, 32, 64, 128]
        self.learning_rates = [0.00001, 0.0001, 0.001, 0.01]  # Client learning rate
        self.momentums = [0.6, 0.7, 0.8, 0.9]
        self.dropouts = [0.0, 0.5]

        # The distribution related parameters
        self.hyperparamconfigs = [self.batch_sizes, self.learning_rates, self.momentums, self.dropouts]
        self.dist = []   
        self.configs = []
        self.currentconfig = []

        self.build_configs()                     # Group hyperparameters into configurations, build an uniform distribution

        # Other newly added parameters
        self.server_lr = 1                       # Federator learning rate
        self.server_gamma = 1 - pow(10, -2)      # Parameter for decreasing server learning rate
        self.entropy = cal_dist_entropy(self.dist) # Entropy of the distribution
        self.entropy_threshold = 4.5            # Threshold in the paper is 0.0001

        self.scheduler_step_size = 50
        self.scheduler_gamma = 1                 # No decay for client learning rate
        self.min_lr = 1e-10

        self.round_worker_selection_strategy = None
        self.round_worker_selection_strategy_kwargs = None

        self.save_model = False
        self.save_temp_model = False
        self.save_epoch_interval = 1
        self.save_model_path = "models"
        self.epoch_save_start_suffix = "start"
        self.epoch_save_end_suffix = "end"
        self.get_poison_effort = 'half'
        self.num_workers = 50
        # self.num_poisoned_workers = 10

        self.rank = 0
        self.world_size = 0
        self.data_sampler = None
        self.distributed = False
        self.available_nets = {
            "Cifar100ResNet" : Cifar100ResNet,
            "Cifar100VGG" : Cifar100VGG,
            "Cifar10CNN" : Cifar10CNN,
            "Cifar10ResNet" : Cifar10ResNet,
            "FashionMNISTCNN" : FashionMNISTCNN,
            "FashionMNISTResNet" : FashionMNISTResNet

        }
        self.net = None
        self.set_net_by_name('FashionMNISTCNN')
        # self.net = FashionMNISTCNN
        # self.net = Cifar100ResNet
        # self.net = FashionMNISTResNet
        # self.net = Cifar10ResNet
        # self.net = Cifar10ResNet
        self.dataset_name = 'fashion-mnist'
        self.train_data_loader_pickle_path = {
            'cifar10': 'data_loaders/cifar10/train_data_loader.pickle',
            'fashion-mnist': 'data_loaders/fashion-mnist/train_data_loader.pickle',
            'cifar100': 'data_loaders/cifar100/train_data_loader.pickle',
        }

        self.test_data_loader_pickle_path = {
            'cifar10': 'data_loaders/cifar10/test_data_loader.pickle',
            'fashion-mnist': 'data_loaders/fashion-mnist/test_data_loader.pickle',
            'cifar100': 'data_loaders/cifar100/test_data_loader.pickle',
        }

        # self.train_data_loader_pickle_path = "data_loaders/cifar10/train_data_loader.pickle"
        # self.test_data_loader_pickle_path = "data_loaders/cifar10/test_data_loader.pickle"

        # self.train_data_loader_pickle_path = "data_loaders/fashion-mnist/train_data_loader.pickle"
        # self.test_data_loader_pickle_path = "data_loaders/fashion-mnist/test_data_loader.pickle"

        # self.train_data_loader_pickle_path = "data_loaders/cifar100/train_data_loader.pickle"
        # self.test_data_loader_pickle_path = "data_loaders/cifar100/test_data_loader.pickle"

        self.loss_function = torch.nn.CrossEntropyLoss

        self.default_model_folder_path = "default_models"

        self.data_path = "data"

    def get_distributed(self):
        return self.distributed

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def set_sampler(self, sampler):
        self.data_sampler = sampler

    def get_sampler(self):
        return self.data_sampler

    def get_round_worker_selection_strategy(self):
        return self.round_worker_selection_strategy

    def get_round_worker_selection_strategy_kwargs(self):
        return self.round_worker_selection_strategy_kwargs

    def set_round_worker_selection_strategy_kwargs(self, kwargs):
        self.round_worker_selection_strategy_kwargs = kwargs

    def set_client_selection_strategy(self, strategy):
        self.round_worker_selection_strategy = strategy

    def get_data_path(self):
        return self.data_path

    def get_epoch_save_start_suffix(self):
        return self.epoch_save_start_suffix

    def get_epoch_save_end_suffix(self):
        return self.epoch_save_end_suffix

    def get_dataloader_list(self):
        return list(self.train_data_loader_pickle_path.keys())

    def get_nets_list(self):
        return list(self.available_nets.keys())


    def set_train_data_loader_pickle_path(self, path, name='cifar10'):
        self.train_data_loader_pickle_path[name] = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path[self.dataset_name]

    def set_test_data_loader_pickle_path(self, path, name='cifar10'):
        self.test_data_loader_pickle_path[name] = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path[self.dataset_name]

    def set_net_by_name(self, name: str):
        self.net = self.available_nets[name]
        # net_dict = {
        #     'cifar10-cnn': Cifar10CNN,
        #     'fashion-mnist-cnn': FashionMNISTCNN,
        #     'cifar100-resnet': Cifar100ResNet,
        #     'fashion-mnist-resnet': FashionMNISTResNet,
        #     'cifar10-resnet': Cifar10ResNet,
        #     'cifar100-vgg': Cifar100VGG,
        # }
        # self.net = net_dict[name]

    def get_cuda(self):
        return self.cuda

    def get_scheduler_step_size(self):
        return self.scheduler_step_size

    def get_scheduler_gamma(self):
        return self.scheduler_gamma

    def get_min_lr(self):
        return self.min_lr

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_num_epochs(self):
        return self.epochs

    def set_num_poisoned_workers(self, num_poisoned_workers):
        self.num_poisoned_workers = num_poisoned_workers

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_model_save_path(self, save_model_path):
        self.save_model_path = save_model_path

    def get_logger(self):
        return self.logger

    def get_loss_function(self):
        return self.loss_function

    def get_net(self):
        return self.net

    def get_num_workers(self):
        return self.num_workers

    def get_num_poisoned_workers(self):
        return self.num_poisoned_workers

    def get_poison_effort(self):
        return self.get_poison_effort

    def get_learning_rate(self):
        return self.lr

    def get_momentum(self):
        return self.momentum

    def get_shuffle(self):
        return self.shuffle

    def get_batch_size(self):
        return self.batch_size

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_log_interval(self):
        return self.log_interval

    def get_save_model_folder_path(self):
        return self.save_model_path

    def get_learning_rate_from_epoch(self, epoch_idx):
        lr = self.lr * (self.scheduler_gamma ** int(epoch_idx / self.scheduler_step_size))

        if lr < self.min_lr:
            self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

            return self.min_lr

        self.logger.debug("LR: {}".format(lr))

        return lr

    def get_contribution_measurement_round(self):
        return  self.contribution_measurement_round

    def get_contribution_measurement_metric(self):
        return self.contribution_measurement_metric

    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        if not self.save_model:
            return False

        if epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0:
            return True

    def build_configs(self):
        dist = []
        configs = []
        for c in self.hyperparamconfigs :
            dist, configs = setup_configs(dist, configs, c)
        self.dist = dist
        self.configs = configs

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Test Batch Size: {}\n".format(self.test_batch_size) + \
               "Epochs: {}\n".format(self.epochs) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Momentum: {}\n".format(self.momentum) + \
               "CUDA Enabled: {}\n".format(self.cuda) + \
               "Shuffle Enabled: {}\n".format(self.shuffle) + \
               "Log Interval: {}\n".format(self.log_interval) + \
               "Scheduler Step Size: {}\n".format(self.scheduler_step_size) + \
               "Scheduler Gamma: {}\n".format(self.scheduler_gamma) + \
               "Scheduler Minimum Learning Rate: {}\n".format(self.min_lr) + \
               "Client Selection Strategy: {}\n".format(self.round_worker_selection_strategy) + \
               "Client Selection Strategy Arguments: {}\n".format(json.dumps(self.round_worker_selection_strategy_kwargs, indent=4, sort_keys=True)) + \
               "Model Saving Enabled: {}\n".format(self.save_model) + \
               "Model Saving Interval: {}\n".format(self.save_epoch_interval) + \
               "Model Saving Path (Relative): {}\n".format(self.save_model_path) + \
               "Epoch Save Start Prefix: {}\n".format(self.epoch_save_start_suffix) + \
               "Epoch Save End Suffix: {}\n".format(self.epoch_save_end_suffix) + \
               "Number of Clients: {}\n".format(self.num_workers) + \
               "Number of Poisoned Clients: {}\n".format(self.num_poisoned_workers) + \
               "NN: {}\n".format(self.net) + \
               "Train Data Loader Path: {}\n".format(self.train_data_loader_pickle_path) + \
               "Test Data Loader Path: {}\n".format(self.test_data_loader_pickle_path) + \
               "Loss Function: {}\n".format(self.loss_function) + \
               "Default Model Folder Path: {}\n".format(self.default_model_folder_path) + \
               "Data Path: {}\n".format(self.data_path) + \
               "Dataset Name: {}\n".format(self.dataset_name)
