from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class TrainingSettings(BaseModel):
    """
        Settings of model for training.
    """

    # path to dataset
    dataset: Optional[str] = Field(None, description="Path to dataset manifest file (yaml).")

    # settings for saving of results
    results_path: str = Field("runs", description="Path for results of training.")
    project: str = Field("detect", description="Name of project of training.")
    resume: bool = Field(False, description="Resumes training from last project if option is `True`.")
    starting_checkpoint: Optional[str] = Field(
        None, description="Path to starting checkpoint of model. If `resume=True`, then option is ignored. "
    )

    model_size: Literal["nano", "small", "medium", "large"] = Field(
        "nano", description="Size of model. One from next options: 'nano', 'small', 'medium', 'large'."
    )

    # training settings
    enable_amp: bool = Field(False, description="Enable AMP (automatic mixed precision).")
    num_workers: int = Field(0, description="Number of threads used in data loaders.")
    epochs: int = Field(10, ge=1, description="Number of Epochs of training.")
    batch_size: int = Field(32, ge=1, description="Size of batch in training.")

    # loss settings
    weight_box: float = Field(7.5, ge=0.0, description="Weight of box loss (>= 0).")
    weight_cls: float = Field(0.5, ge=0.0, description="Weight of cls loss (>= 0).")
    weight_dfl: float = Field(1.5, ge=0.0, description="Weight of dfl loss (>= 0).")

    # optimizer settings
    weight_decay: float = Field(0.0005, ge=0.0, description="Weight decay (>= 0).")
    min_lr: float = Field(0.0001, ge=0.0, description="Minimal learning rate (>= 0).")
    max_lr: float = Field(0.01, ge=0.0, description="Maximal learning rate (>= 0).")
    momentum: float = Field(0.937, ge=0.0, le=1.0, description="Momentum in optimizer (0-1).")
    warmup_epochs: float = Field(3.0, ge=0.0, description="Number of warmup epochs (>= 0).")

    names: List[str] = Field(..., description="List of names of classes (labels).")

    # augmentations settings
    mosaic: float = Field(0.0, ge=0.0, le=1.0, description="Probability of mosaic augmentation (0-1).")
    adversarial: float = Field(0.0, ge=0.0, le=1.0, description="Probability of adversarial batch (0-1).")

    class Config:
        str_strip_whitespace = True
        validate_assignment = True  # валидация при присвоении после создания
