from typing import List, Dict

import csv
import os
import warnings

import torch
import tqdm
import yaml
from torch.utils import data
from torchvision.transforms.v2 import Compose

from nets import nn
from utils import util
from nets.criterion import ComputeLoss
from utils.dataset import Dataset
from utils.logging import Logger
import utils.transforms as transforms
from utils.settings import TrainingSettings
import shutil
from torchinfo import summary
import numpy as np

warnings.filterwarnings("ignore")

def first_available(dir_path:str, subdir:str) -> int:
    """
        Finds first free number of run.
    """
    counter = 1
    while True:
        # create potential path
        potential_subdir_name = f"{subdir}{counter}"
        full_path = os.path.join(dir_path, potential_subdir_name)

        # check path exists
        if not os.path.exists(full_path):
            return counter  # return free number
        
        counter += 1

class Trainer:
    
    # length on one signal sample
    REGULAR_LENGTH = 50000
    # length of network input
    INPUT_LENGTH = 49152
    
    def __init__(self, args: Dict):
        self._args = args
        
        self._chkpt_path = self._args.get("starting_checkpoint", None) 
        if self._chkpt_path:
            self._chkpt_path = os.path.abspath(self._chkpt_path)   
        # parameters of 
        self._results_path = os.path.abspath(self._args.get("results_path", "runs"))
        self._project = self._args.get("project", "detect")
        self._num = first_available(self._results_path, self._project)
        # is resume available
        self._resume = self._args.get("resume", False) and self._num > 1
        # if need to resume training - load last session
        if self._resume:
            self._num = max(self._num - 1, 1)
        # path for results    
        self._project_path = os.path.join(self._results_path, f"{self._project}{self._num}")
        # load settings
        self._load_settings()
        self._init_model()  
        self._prepare_transforms()
        self._init_dataloaders()
        self._init_training()
        self._load_model_weights()
        
    
    def _load_settings(self):
        """ Loads and saves configuration of training. """
        # load settings
        if self._resume:
            args_path = os.path.join(self._project_path, "args.yaml")
        else:
            args_path = "./utils/args.yaml"
        # read the config
        with open(args_path, errors='ignore') as f:
            params = yaml.safe_load(f)
        # merge parameters
        self._args.update(params)
        # parse settings
        self._settings = TrainingSettings.from_dict(self._args)
        self._num_classes = len(self._settings.names)
    
    def _get_model(self):
        """ Creates a model of desired size."""
        if self._settings.model_size == "nano":
            return nn.yolo_v11_n(self._num_classes)
        if self._settings.model_size == "small":
            return nn.yolo_v11_s(self._num_classes)
        if self._settings.model_size == "medium":
            return nn.yolo_v11_m(self._num_classes)
        if self._settings.model_size == "large":
            return nn.yolo_v11_l(self._num_classes)
    
    def _init_model(self):
        """
            Initializes a model.
        """
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        # create model
        self._model = self._get_model()
        self._model.to(self._device)
        
        # Exponential moving average for weights of model
        self._ema = util.EMA(self._model)
    
    def _init_dataloaders(self):
        # load dataset
        self._train_dataset = Dataset(self._settings.dataset, transform=self._train_transform, type="train", mosaic_prob=self._settings.mosaic)
        self._val_dataset = Dataset(self._settings.dataset, transform=self._test_transform, type="val")
        self._test_dataset = Dataset(self._settings.dataset, transform=self._test_transform, type="test")
         
        # create dataloaders
        self._train_loader = data.DataLoader(self._train_dataset, self._settings.batch_size, 
                                num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)        
        self._val_loader = data.DataLoader(self._val_dataset, self._settings.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, collate_fn=Dataset.collate_fn)
        self._test_loader = data.DataLoader(self._test_dataset, self._settings.batch_size, shuffle=False,
                                num_workers=4, pin_memory=True, collate_fn=Dataset.collate_fn)

    def _init_training(self):
        """
            Initializes everything needed for training.
        """

        # scaling of Regularization to make it independent of batch_size
        self._accumulate = max(round(64 / (self._settings.batch_size)), 1)
        self._settings.weight_decay *= self._settings.batch_size * self._accumulate / 64
        # create optimizer
        self._optimizer = torch.optim.SGD(util.set_params(self._model, self._settings.weight_decay),
                                self._settings.min_lr, self._settings.momentum, nesterov=True)
        self._start_epoch = 0
        
        

        # Scheduler
        self._num_steps = len(self._train_loader)
        self._scheduler = util.LinearLR(self._settings.min_lr, self._settings.max_lr,
                                        self._settings.warmup_epochs, self._settings.epochs, self._num_steps)

        self._best_score = 0
        # automatic mixed precision scaling of gradients
        self._amp_scale = torch.amp.GradScaler() #if self._settings.enable_amp else None
        # loss function
        self._criterion = ComputeLoss(self._model, self._settings.weight_box,
                                          self._settings.weight_cls, self._settings.weight_dfl)
        
    def _prepare_transforms(self):
        """
            Creates transforms for dataLoaders.
        """
        # TODO: read parameters
        self._train_transform = Compose([
            transforms.MarkScaling(300),        # apply scaling to data in bboxes
            transforms.RandomResize(0.1),       # stretch or compress signal in time domain
            transforms.PhaseSpectrumNoise(0.5), # adds noise in phase spectrum
            transforms.RandomJitter(0.6),       # add random normal noise
            transforms.TimeMasking(0.2),        # erase random area in time domain
            transforms.RandomCrop(Trainer.INPUT_LENGTH),    # make a random crop to desired size
            transforms.RandomInvert(0.5),       # invert sign of channels
            transforms.RandomReverse(0.5),      # reverse signal in time domain
            transforms.ChannelScaling(),        # apply random scaling to channels
            transforms.ChannelMasking(0.1),     # erase random channels
            ])
        self._test_transform = Compose([
            transforms.Padding(Trainer.REGULAR_LENGTH), # add padding of signal to full size
            transforms.Resize(Trainer.INPUT_LENGTH),    # apply resizing to signal
            ])
        
    def _load_model_weights(self):
        # paths to resulting files
        self._best_path = os.path.join(self._project_path, 'best.pt')
        self._last_path = os.path.join(self._project_path, 'last.pt')
        
        checkpoint: Dict = None
        # load state of model for resume
        if self._resume and os.path.exists(self._last_path):
            checkpoint = torch.load(self._last_path, weights_only=False)
        # start from checkpoint
        if not self._resume and self._chkpt_path:
            checkpoint = torch.load(self._chkpt_path, weights_only=False)
        # load checkpoint
        if checkpoint is not None:
            self._model.load_state_dict(checkpoint['model'], strict=False)   # load model
            self._start_epoch = checkpoint['epoch']                  # load number of last epoch
            self._ema.load_state_dict(checkpoint['ema'])             # load Averaged weights of model (for testing)
            if "optimizer" in checkpoint.keys():
                self._optimizer.load_state_dict(checkpoint["optimizer"]) # load optimizer state
    
    def _strip_optimizer(self, name):
        chkpt = torch.load(name, weights_only=False)
        if "optimizer" in chkpt.keys():
            del chkpt["optimizer"]
        torch.save(chkpt, f=name)
    
    def train(self):
        # create folder for results
        os.makedirs(self._project_path, exist_ok=True)
        # save training config
        with open(os.path.join(self._project_path, "args.yaml"), "w") as f:
            yaml.dump(self._args, f)
        
        best = 0
        self._logger_path = os.path.join(self._project_path, 'step.csv')
        # show summary
        dummy_size = (self._settings.batch_size, 30, self.REGULAR_LENGTH)
        print(summary(self._model, dummy_size))
        with open(self._logger_path, "a") as log:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                    'box_train', 'cls_train', 'dfl_train',
                                    'box_val', 'cls_val', 'dfl_val',
                                    'Recall', 'Precision', 'mAP@50', 'mAP'])
            if not self._resume:
                logger.writeheader()
            
            for epoch in range(self._start_epoch, self._settings.epochs):
                self._epoch_results = dict()
                self._model.train()
                # turn off mosaic augmentations at last 10 epochs
                if self._settings.epochs - epoch == 10:
                    self._train_loader.dataset.mosaic_prob = 0
                    self._settings.adversal = 0
                # size of progress bar
                p_bar = enumerate(self._train_loader)

                # init progress bar with header
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=self._num_steps)

                self._optimizer.zero_grad()
                # averaging of loss on dataset
                avg_box_loss = util.AverageMeter()
                avg_cls_loss = util.AverageMeter()
                avg_dfl_loss = util.AverageMeter()
                for i, (samples, targets) in p_bar:
                    
                    step = i + self._num_steps * epoch
                    self._scheduler.step(step, self._optimizer)
                    # convert training data
                    samples = samples.to(self._device).float()
                    if np.random.uniform(0,1) < self._settings.adversal:
                        samples = util.get_adversal_examples(self._model, self._criterion,
                                            self._amp_scale, samples, targets)

                    # Forward
                    # if self._settings.enable_amp:
                    with torch.amp.autocast('cuda'):
                        outputs = self._model(samples)  # forward
                        loss_box, loss_cls, loss_dfl = self._criterion(outputs, targets)
                    # else:
                    #     loss_box, loss_cls, loss_dfl = self._criterion(outputs, targets)
                    # update average loss
                    avg_box_loss.update(loss_box.item(), samples.size(0))
                    avg_cls_loss.update(loss_cls.item(), samples.size(0))
                    avg_dfl_loss.update(loss_dfl.item(), samples.size(0))
                    # scale loss by batch_size
                    loss_box *= self._settings.batch_size
                    loss_cls *= self._settings.batch_size
                    loss_dfl *= self._settings.batch_size

                    # Backward
                    self._amp_scale.scale(loss_box + loss_cls + loss_dfl).backward()

                    # Optimize
                    if step % self._accumulate == 0:
                        self._amp_scale.unscale_(self._optimizer)  # unscale gradients
                        util.clip_gradients(self._model)  # clip gradients
                        self._amp_scale.step(self._optimizer)
                        self._amp_scale.update()
                        self._optimizer.zero_grad()
                        if self._ema:
                            self._ema.update(self._model)

                    torch.cuda.synchronize()

                    # Logging
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{self._settings.epochs}', memory,
                                                    avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

                self._epoch_results["box_train"] = avg_box_loss.avg
                self._epoch_results["cls_train"] = avg_cls_loss.avg
                self._epoch_results["dfl_train"] = avg_dfl_loss.avg
                # validation
                self.validate()
                # logging results
                logger.writerow({
                        'epoch': str(epoch + 1).zfill(3),
                        **{k: f'{v:.3f}' for k, v in self._epoch_results.items()}
                    })
                log.flush()

                # create checkpoint
                save =  {
                            'epoch': epoch + 1,
                            'model': self._model.state_dict(),
                            'ema': self._ema.state_dict(),
                            'optimizer': self._optimizer.state_dict()
                        }
                # save last model
                torch.save(save, f=self._last_path)
                
                # Update best mAP and save best model
                if self._epoch_results.get("mAP", 0) > best:
                    best = self._epoch_results.get("mAP", 0)
                    shutil.copy(self._last_path, self._best_path)
                # delete checkpoint
                del save
        # delete optimizer from checkpoints
        self._strip_optimizer(self._last_path)
        self._strip_optimizer(self._best_path)
    
    @torch.no_grad()
    def validate(self):

        self._ema.ema.eval()
        self._ema.ema.set_validate(True)
        
        criterion = ComputeLoss(self._ema.ema, self._settings.weight_box,
                        self._settings.weight_cls, self._settings.weight_dfl)
        
        # Configure
        iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()

        metrics = []
        p_bar = tqdm.tqdm(self._train_loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
        # Sum of losses on validation 
        avg_box_loss = util.AverageMeter()
        avg_cls_loss = util.AverageMeter()
        avg_dfl_loss = util.AverageMeter()
        for samples, targets in p_bar:
            samples = samples.cuda().float()
            w = samples.shape[2]  # length of signal
            scale = torch.tensor((w, w)).cuda()
            # Inference
            raw, outputs = self._ema.ema(samples)
            # Calculate loss
            loss_box, loss_cls, loss_dfl = criterion(raw, targets)
            # update average losses
            avg_box_loss.update(loss_box.item(), samples.size(0))
            avg_cls_loss.update(loss_cls.item(), samples.size(0))
            avg_dfl_loss.update(loss_dfl.item(), samples.size(0))
            # NMS
            outputs = util.non_max_suppression(outputs)
            # Metrics
            for i, output in enumerate(outputs):
                idx = targets['idx'] == i
                cls = targets['cls'][idx]
                box = targets['box'][idx]

                cls = cls.cuda()
                box = box.cuda()

                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

                if output.shape[0] == 0:
                    if cls.shape[0]:
                        metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                    continue
                # Evaluate
                if cls.shape[0]:
                    target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                    metric = util.compute_metric(output[:, :4], target, iou_v)
                # Append
                metrics.append((metric, output[:, 2], output[:, 3], cls.squeeze(-1)))
        self._ema.ema.set_validate(False)
        # Compute metrics
        metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
        r = self._epoch_results
        if len(metrics) and metrics[0].any():
            tp, fp, r["Precision"], r["Recall"],  \
                            r["mAP@50"], r["mAP"] = util.compute_ap(*metrics)
        # Print results
        print(('%10s' + '%10.3g' * 4) % ('', r.get("Precision",0), r.get("Recall",0), r.get("mAP@50",0), r.get("mAP",0)))
        # Return results
        r["box_val"] = avg_box_loss.avg
        r["cls_val"] = avg_cls_loss.avg
        r["dfl_val"] = avg_dfl_loss.avg
        
    @torch.no_grad()
    def test(self):
        self._ema.ema.eval()
        
        # Configure
        iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()

        metrics = []
        p_bar = tqdm.tqdm(self._train_loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))

        for samples, targets in p_bar:
            samples = samples.cuda().float()
            w = samples.shape[2]  # length of signal
            scale = torch.tensor((w, w)).cuda()
            # Inference
            outputs = self._ema.ema(samples)
            # NMS
            outputs = util.non_max_suppression(outputs)
            # Metrics
            for i, output in enumerate(outputs):
                idx = targets['idx'] == i
                cls = targets['cls'][idx]
                box = targets['box'][idx]

                cls = cls.cuda()
                box = box.cuda()

                metric = torch.zeros(output.shape[0], n_iou, dtype=torch.bool).cuda()

                if output.shape[0] == 0:
                    if cls.shape[0]:
                        metrics.append((metric, *torch.zeros((2, 0)).cuda(), cls.squeeze(-1)))
                    continue
                # Evaluate
                if cls.shape[0]:
                    target = torch.cat(tensors=(cls, util.wh2xy(box) * scale), dim=1)
                    metric = util.compute_metric(output[:, :4], target, iou_v)
                # Append
                metrics.append((metric, output[:, 2], output[:, 3], cls.squeeze(-1)))
        # Compute metrics
        metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
        if len(metrics) and metrics[0].any():
            tp, fp, Precision, Recall,  \
                            mAP50, mAP = util.compute_ap(*metrics)
        # Print results
        print(('%10s' + '%10.3g' * 4) % ('', Precision, Recall, mAP50, mAP))
    
    @torch.no_grad()
    def predict(self, data: torch.Tensor, conf: float = 0.1):
        data = self._test_transform(data)
        outputs = self._ema.ema(data)
        outputs = util.non_max_suppression(outputs, conf)
        return outputs
        
        

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
    Logger.add_console_handler()
    main()
