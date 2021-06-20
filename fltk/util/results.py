from dataclasses import dataclass
from typing import Any

@dataclass
class EpochData:
    epoch_id: int
    duration_train: int
    duration_test: int
    loss_train: float
    accuracy: float
    loss: float
    class_precision: Any
    class_recall: Any
    batch_size: int
    test_datasize: int
    dist: Any
    chosen_config_index: int # index of chosen configuration tuple in configs/dist
    client_id: str = None

    def to_csv_line(self):
        delimeter = ','
        values = self.__dict__.values()
        values = [str(x) for x in values]
        return delimeter.join(values)
