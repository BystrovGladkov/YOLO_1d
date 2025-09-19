import torch

import yolo_1d.utils.util as util
from yolo_1d import Trainer

def main():
    
    config = {
        "project": "small",
        "results_path": "runs",
        "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
        "batch_size": 50,
        "epochs": 200,
        "resume": True,
        "model_size": "small",
        "enable_amp": True,
        # "starting_checkpoint": r"E:\git\neuro\YOLOv11-pt\runs\large2\best.pt",
    }

    util.setup_seed()
    util.setup_multi_processes()

    trainer = Trainer(config)
    
    trainer.train()
    # trainer.validate()
    print("Testing:")
    trainer.test()

    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()