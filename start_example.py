import torch

import yolo_1d.utils.util as util
from yolo_1d import Trainer, TrainingSettings

def main():
    
    config = {
        "project": "v0.5_large_no_occlusion",
        "results_path": "runs",
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        "batch_size": 50,
        "epochs": 500,
        "resume": True,
        "model_size": "large",
        "enable_amp": True,
        "early_stopping_epochs": 20,
        "names":["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
        # "starting_checkpoint": r"E:\git\neuro\YOLO_1d\runs\large_data4\best.pt",
        "weight_box": 10,
        "weight_dfl": 3,
        "num_workers": 4,
        "augmentations":{
            "regular_length": 50000, 
            "input_length": 49152, 
            "random_resize": 0.1, # geometric
            "random_reverse": 0.5, # geometric 
            "random_invert": 0.5, # amplitude
            "channel_scaling": 0.2, # amplitude
            "time_masking": 0, # occlision
            "channel_masking": 0, # occlision
            "phase_spectrum_noise": 0.5, # spectrun and noise
            "random_jitter": 0.6, # spectrun and noise
            "mosaic": 0.5, # composite
            "mix_up": 0.5, # composite
            "adversarial": 0, # adversal
        }
    }

    util.setup_seed()
    util.setup_multi_processes()

    t = TrainingSettings(**config)
    
    trainer = Trainer(t)
    
    trainer.train()
    trainer.validate()
    print("Testing:")
    trainer.test()

    torch.cuda.empty_cache()

def test():
    config = {
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        "batch_size": 50,
        "epochs": 300,
        # "resume": True,
        "model_size": "large",
        "enable_amp": True,
        "names":["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
        "starting_checkpoint": r"E:\git\neuro\YOLO_1d\runs\v0.4_large1\best.pt",
    }

    util.setup_seed()
    util.setup_multi_processes()

    t = TrainingSettings(**config)
    
    trainer = Trainer(t)
    
    # trainer.train()
    # trainer.validate()
    print("Testing:")
    trainer.test()

    torch.cuda.empty_cache()

def export():
    config = {
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        "batch_size": 50,
        "model_size": "small",
        "enable_amp": True,
        "names":["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
        "starting_checkpoint": r"E:\git\neuro\YOLO_1d\runs\training_08_121\best.pt",
    }

    t = TrainingSettings(**config)
    t.names
    
    trainer = Trainer(t)
    # trainer.validate()
    # trainer.test()
    print(trainer._model.head.stride)
    
    # trainer.export_onnx("YOLO_1d.onnx")

if __name__ == "__main__":
    main()
    # test()
    # export()