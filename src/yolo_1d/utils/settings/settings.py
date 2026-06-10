from typing import List, Literal, Optional
from pydantic import BaseModel, Field

class AugmentationSettings(BaseModel):
    regular_length: int = Field(50000, description="regular length of input signal, used to save aspect in validation transforms.")
    input_length: int = Field(49152, description="Target input length of signal.")
    
    # параметры для train transform
    mark_scaling: int = Field(300, ge=1, description="Scale used in MarkScaling.")
    random_resize: float = Field(0.1, ge=0.0, description="RandomResize strength.")
    
    phase_spectrum_noise: float = Field(0.5, ge=0.0, description="Strength of phase spectrum noise.")
    random_jitter: float = Field(0.6, ge=0.0, description="Strength of jitter.")
    random_invert: float = Field(0.5, ge=0.0, le=1.0, description="Probability of sign inversion.")
    random_reverse: float = Field(0.5, ge=0.0, le=1.0, description="Probability of time reversal.")
    channel_scaling: float = Field(0.4, ge=0.0, description="Strength of channel scaling.")
    time_masking: float = Field(0.2, ge=0.0, le=1.0, description="Max length of time masking.")
    channel_masking: float = Field(0.1, ge=0.0, le=1.0, description="Probability of channel masking.")

    # настройки для mixup / mosaic / adversarial
    mosaic: float = Field(0.0, ge=0.0, le=1.0, description="Probability of mosaic augmentation.")
    mix_up: float = Field(0.0, ge=0.0, le=1.0, description="Probability of mixup augmentation.")
    mix_up_alpha: float = Field(20.0, ge=0.0, description="Beta distribution alpha for mixup.")
    adversarial: float = Field(0.0, ge=0.0, le=1.0, description="Probability of adversarial batch.")

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
    early_stopping_epochs: int = Field(0, ge=0, description="Number of epochs for early stop of training.")

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
    augmentations: AugmentationSettings = Field(default_factory=AugmentationSettings)

    class Config:
        str_strip_whitespace = True
        validate_assignment = True  # валидация при присвоении после создания
