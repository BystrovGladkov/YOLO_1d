from typing import List, Dict

import csv
import os
from pathlib import Path
import warnings
import shutil

import torch
import tqdm
import yaml
from torch.utils import data
from torchvision.transforms.v2 import Compose
from torchinfo import summary
import numpy as np
import pandas as pd

from yolo_1d.nets import nn
from yolo_1d.utils import util
from yolo_1d.nets.criterion import ComputeLoss
from yolo_1d.utils.dataset import Dataset
from yolo_1d.utils.transforms import build_training_transforms, build_validation_transform
from yolo_1d.utils.settings import TrainingSettings, get_default_settings, save_settings, read_settings
from yolo_1d.utils.constants import *

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
    
    def __init__(self, settings: TrainingSettings=None):
        self._settings = settings
        
        self._chkpt_path = self._settings.starting_checkpoint 
        # parameters of 
        self._results_path = os.path.abspath(self._settings.results_path)
        self._project = self._settings.project
        self._num = first_available(self._results_path, self._project)
        # is resume available
        self._resume = self._settings.resume and self._num > 1
        # if need to resume training - load last session
        if self._resume:
            self._num = max(self._num - 1, 1)
        # path for results    
        self._project_path = os.path.join(self._results_path, f"{self._project}{self._num}")
        self._num_classes = len(self._settings.names)
        self._epoch_results = dict()
        # load settings
        self._init_model()  
        self._prepare_transforms()
        self._load_model_weights()
        
    def _get_model(self):
        """ Creates a model of desired size."""
        return nn.get_model(self._settings.model_size, self._num_classes,
                            input_channels=self._settings.input_channels,
                            input_length=self._settings.augmentations.input_length)
        
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
        # if fields already initialized - exit
        if hasattr(self, "_train_dataset"):
            return
        # load dataset
        self._train_dataset = Dataset(self._settings.dataset, transform=self._train_transform, type="train", 
                                      mosaic_prob=self._settings.augmentations.mosaic,
                                      mix_up_prob=self._settings.augmentations.mix_up,
                                      mix_up_alpha=self._settings.augmentations.mix_up_alpha)
        self._val_dataset = Dataset(self._settings.dataset, transform=self._test_transform, type="val")
        self._test_dataset = Dataset(self._settings.dataset, transform=self._test_transform, type="test")
        
        weights = util.get_data_weights(self._train_dataset)
        sampler = torch.utils.data.WeightedRandomSampler(weights=weights,
                                    num_samples=self._train_dataset.n, replacement=True)
        
        # create dataloaders
        self._train_loader = data.DataLoader(self._train_dataset, self._settings.batch_size, 
                                num_workers=self._settings.num_workers, pin_memory=True, collate_fn=Dataset.collate_fn, sampler=sampler)        
        self._val_loader = data.DataLoader(self._val_dataset, self._settings.batch_size, shuffle=False,
                                num_workers=self._settings.num_workers, pin_memory=True, collate_fn=Dataset.collate_fn)
        self._test_loader = data.DataLoader(self._test_dataset, self._settings.batch_size, shuffle=False,
                                num_workers=self._settings.num_workers, pin_memory=True, collate_fn=Dataset.collate_fn)

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
        # if need to resume training
        if self._resume and self._checkpoint and "optimizer" in self._checkpoint.keys():
            self._optimizer.load_state_dict(self._checkpoint["optimizer"]) # load optimizer state
        if not self._resume:
            self._start_epoch = 0
        
        # Scheduler
        self._num_steps = len(self._train_loader)
        self._scheduler = util.LinearLR(self._settings.min_lr, self._settings.max_lr,
                                        self._settings.warmup_epochs, self._settings.epochs, self._num_steps)
        
        # automatic mixed precision scaling of gradients
        self._amp_scale = torch.amp.GradScaler() #if self._settings.enable_amp else None
        # loss function
        self._criterion = ComputeLoss(self._model, self._settings.weight_box,
                                          self._settings.weight_cls, self._settings.weight_dfl)
        
    def _prepare_transforms(self):
        """
            Creates transforms for dataLoaders.
        """
        self._train_transform = build_training_transforms(self._settings.augmentations)
        # self._train_transform = build_validation_transform(self._settings.augmentations)
        self._test_transform = build_validation_transform(self._settings.augmentations)
        
    def _load_model_weights(self):
        # paths to resulting files
        self._best_path = os.path.join(self._project_path, 'best.pt')
        self._last_path = os.path.join(self._project_path, 'last.pt')
        
        self._checkpoint: Dict = None
        # load state of model for resume
        if self._resume and os.path.exists(self._last_path):
            self._checkpoint = torch.load(self._last_path, weights_only=False)
        # start from checkpoint
        if not self._resume and self._chkpt_path:
            self._checkpoint = torch.load(self._chkpt_path, weights_only=False)
        # load checkpoint
        if self._checkpoint is not None:
            self._model.load_state_dict(self._checkpoint['model'], strict=False)   # load model
            self._start_epoch = self._checkpoint['epoch']                  # load number of last epoch
            self._ema.load_state_dict(self._checkpoint['ema'])             # load Averaged weights of model (for testing)
             
    def _strip_optimizer(self, name):
        chkpt = torch.load(name, weights_only=False)
        if "optimizer" in chkpt.keys():
            del chkpt["optimizer"]
        torch.save(chkpt, f=name)
    
    def _load_best_map(self):
        if self._resume and os.path.exists(self._logger_path):
            df = pd.read_csv(self._logger_path)
            best_idx = df["mAP"].argmax()
            return df["mAP"].max(), df.iloc[best_idx]["epoch"]
        return 0.0, -1
    
    def _load_best_val_loss(self):
        if self._resume and os.path.exists(self._logger_path):
            df = pd.read_csv(self._logger_path)
            loss = df["box_val"] + df["cls_val"] + df["dfl_val"]
            best_idx = loss.argmin()
            return loss.min(), df.iloc[best_idx]["epoch"]
        return np.inf, -1
    
    def train(self):
        self._init_dataloaders()
        self._init_training()
        # create folder for results
        os.makedirs(self._project_path, exist_ok=True)
        # save training config
        save_settings(os.path.join(self._project_path, "args.yaml"), self._settings)
        
        self._adversarial = self._settings.augmentations.adversarial
        # show summary
        dummy_size = (self._settings.batch_size, self._settings.input_channels,
                      self._settings.augmentations.input_length)
        print(summary(self._model, dummy_size))
        
        self._logger_path = os.path.join(self._project_path, 'step.csv')
        # best, best_epoch = self._load_best_map()
        best, best_epoch = self._load_best_val_loss()
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
                if self._settings.epochs - epoch < 10:
                    self._train_loader.dataset.mosaic_prob = 0
                    self._adversarial = 0
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
                    if np.random.uniform(0,1) < self._adversarial:
                        samples = util.get_adversal_examples(self._model, self._criterion,
                                            self._amp_scale, samples, targets)

                    # Forward
                    # if self._settings.enable_amp: TODO: fix amp
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
                
                new_val_loss = self._epoch_results["dfl_val"] + self._epoch_results["cls_val"] + self._epoch_results["box_val"]
                # Update best mAP and save best model
                # if self._epoch_results.get("mAP", 0) > best:
                if new_val_loss < best:
                    best = new_val_loss
                    # best = self._epoch_results.get("mAP", 0)
                    best_epoch = epoch
                    shutil.copy(self._last_path, self._best_path)
                # delete checkpoint
                del save
                if (self._settings.early_stopping_epochs > 0) and (epoch - best_epoch > self._settings.early_stopping_epochs):
                    print(f"Early stopping at epoch {epoch}.")
                    break
        # delete optimizer from checkpoints
        self._strip_optimizer(self._last_path)
        self._strip_optimizer(self._best_path)
    
    @torch.no_grad()
    def validate(self):
        self._init_dataloaders()
        
        self._ema.ema.eval()
        self._ema.ema.set_validate(True)
        
        criterion = ComputeLoss(self._ema.ema, self._settings.weight_box,
                        self._settings.weight_cls, self._settings.weight_dfl)
        
        # Configure
        iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()

        metrics = []
        p_bar = tqdm.tqdm(self._val_loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
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
            # get the class-wise Precision and REcall
            _,_, rec, pre, AP50, AP = util.compute_ap(*metrics)
            r["Precision"], r["Recall"], r["mAP@50"], r["mAP"] = rec.mean(), pre.mean(), AP50.mean(), AP.mean()
        else:
            rec = np.zeros(self._num_classes)
            pre = np.zeros(self._num_classes)
            AP50 = np.zeros(self._num_classes)
            AP = np.zeros(self._num_classes)
        # Print results
        print(('%10s' + '%10.3g' * 4) % ('', r.get("Precision",0), r.get("Recall",0), r.get("mAP@50",0), r.get("mAP",0)))
        
        for name, prec, recl, ap50, ap in zip(self._settings.names, rec, pre, AP50, AP):
            print(('%10s' + '%10.3g' * 4) % (name, prec, recl, ap50, ap))
        
        # Return results
        r["box_val"] = avg_box_loss.avg
        r["cls_val"] = avg_cls_loss.avg
        r["dfl_val"] = avg_dfl_loss.avg
        
    @torch.no_grad()
    def test(self):
        self._init_dataloaders()
        self._ema.ema.eval()
        
        # Configure
        iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
        n_iou = iou_v.numel()

        metrics = []
        p_bar = tqdm.tqdm(self._test_loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))

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
        r = dict()
        if len(metrics) and metrics[0].any():
            # get the class-wise Precision and REcall
            _, _, rec, pre, AP50, AP = util.compute_ap(*metrics)
            r["Precision"], r["Recall"], r["mAP@50"], r["mAP"] = rec.mean(), pre.mean(), AP50.mean(), AP.mean()
        else:
            rec = np.zeros(self._num_classes)
            pre = np.zeros(self._num_classes)
            AP50 = np.zeros(self._num_classes)
            AP = np.zeros(self._num_classes)
        # Print results
        print(('%10s' + '%10.3g' * 4) % ('', r.get("Precision",0), r.get("Recall",0), r.get("mAP@50",0), r.get("mAP",0)))
        
        for name, prec, recl, ap50, ap in zip(self._settings.names, rec, pre, AP50, AP):
            print(('%10s' + '%10.3g' * 4) % (name, prec, recl, ap50, ap))
    
    @torch.no_grad()
    def predict(self, data: torch.Tensor, conf: float = 0.1) -> torch.Tensor:
        # create empty boxes and labels to pass into transform
        bboxes, labels = torch.zeros((0,2), device=data.device, dtype=torch.float), torch.zeros((0), device=data.device, dtype=torch.float)
        # convert data to float
        data = data.float()
        # apply transform
        data, _, _ = self._test_transform(data, bboxes, labels)
        # add batch
        if data.ndim == 2:
            data = data.unsqueeze(0)
        data = data.to(self._device)
        # get predictions
        outputs = self._ema.ema(data)
        outputs = util.non_max_suppression(outputs, conf)
        aug = self._settings.augmentations
        if aug.validation_resize:
            # вход был дополнен до regular_length и сжат до input_length — возвращаем масштаб
            for out in outputs:
                out[:, :2] = out[:, :2] / aug.input_length * aug.regular_length
        # без ресайза (ЭКГ) паддинг добавлен справа: координаты уже в отсчётах исходного сигнала
        return outputs
    
    def export_onnx(self, filename, opset=17):
        from yolo_1d.nets.export import ExportWrapper
        aug = self._settings.augmentations
        example_input = (torch.randn((1, self._settings.input_channels, aug.regular_length), device=self._device),)
        self._ema.ema.eval()
        wrapper = ExportWrapper(self._ema.ema, input_length=aug.input_length, resize=aug.validation_resize)
        torch.onnx.export(wrapper, example_input,
            f=filename,   
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            opset_version=opset)
        


