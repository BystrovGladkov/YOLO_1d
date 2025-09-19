from typing import Literal, List, Dict
from dataclasses import dataclass, asdict, field

@dataclass
class TrainingSettings:
    # path to dataset
    dataset: str = None
    # settings for saving of results
    results_path: str = "runs"
    project: str = "detect"
    resume: bool = False # is resume of training needed
    starting_checkpoint: str = None # path to checkpoint to start training from, ignored if `resume` is true
    
    model_size: Literal["nano", "small", "medium", "large"] = "nano"    # size of model
    
    # training settings
    enable_amp: bool = False
    epochs: int = 10
    batch_size: int = 32

    # loss settings
    weight_box: float = 1.0 # weight of box loss
    weight_cls: float = 1.0 # weight of cls loss
    weight_dfl: float = 1.0 # weight of dfl loss

    # optimizer settings
    weight_decay: float = 0.0
    min_lr: float = 0.0001
    max_lr: float = 0.0100
    momentum: float = 0.0
    warmup_epochs: float = 0
    
    names: List[str] = field(default_factory=list)
    # augmentations settings
    mosaic: float = 0.0     # probability of mosaic augmentation
    adversal: float = 0.0   # probability of adversal batch

    @classmethod
    def from_dict(cls, data: dict):
        # Фильтруем словарь, оставляя только те ключи, которые есть в классе
        valid_keys = {key for key in cls.__dataclass_fields__}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

    