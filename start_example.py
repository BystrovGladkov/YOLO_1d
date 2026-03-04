import torch

import yolo_1d.utils.util as util
from yolo_1d import Trainer, TrainingSettings

def main():
    
    config = {
        "project": "training_08_12",
        "results_path": "runs",
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        "batch_size": 50,
        "epochs": 800,
        "resume": False,
        "model_size": "small",
        "enable_amp": True,
        "names":["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
        # "starting_checkpoint": r"E:\git\neuro\YOLOv11-pt\runs\large_data4\best.pt",
        "weight_box": 10,
        "weight_dfl": 3,
    }

    util.setup_seed()
    util.setup_multi_processes()

    t = TrainingSettings(**config)
    t.names
    
    trainer = Trainer(t)
    
    trainer.train()
    trainer.validate()
    print("Testing:")
    # trainer.test()

    torch.cuda.empty_cache()
    # повысить dfl и box

def export():
    config = {
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        "batch_size": 50,
        "model_size": "small",
        "enable_amp": True,
        "names":["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
        "starting_checkpoint": r"E:\git\neuro\YOLOv11-pt\runs\training_08_121\best.pt",
    }
    # config = {
    #     "model_size": "medium",
    #     "enable_amp": True,
    #     "names":["flashbutt", "alumin", "rail end", "axlec", "crossing", "manual arc"],
    #     "starting_checkpoint": r"E:\git\neuro\YOLOv11-pt\runs\medium_new_data1\best.pt",
    # }



    t = TrainingSettings(**config)
    t.names
    
    trainer = Trainer(t)
    trainer.test()
    
    # trainer.export_onnx("YOLO_1d.onnx")

if __name__ == "__main__":
    # main()
    export()