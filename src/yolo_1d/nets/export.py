import torch
from torch.nn import Module

from yolo_1d.utils.constants import *

class ExportWrapper(Module):
    """
        Wrapper over the model to use in exported version.
        
        Performs downsampling interpolation to desired length as preprocessing step. 
        
        Performs scaling of predicted box to original coordinates as postprocessing step.
    """
    
    def __init__(self, model):
        super().__init__() 
        self.model = model
        model.eval()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_length = x.size(dim=-1)
        # perform interpolation to desired length
        x = torch.nn.functional.interpolate(x, INPUT_LENGTH, mode='linear')
        # get predictions
        output = self.model(x)
        # scale the box size
        output[:, :2, :] = output[:, :2, :] / INPUT_LENGTH * input_length
        return output
        