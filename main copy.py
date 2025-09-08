import copy
import csv
import os
import warnings
from argparse import ArgumentParser

import torch
import tqdm
import yaml
from torch.utils import data
from torchvision.transforms.v2 import Compose

from nets import nn
from utils import util
from utils.dataset import Dataset
from utils.logging import Logger
import utils.transforms as transforms

warnings.filterwarnings("ignore")

def first_avaliable(dir_path:str, subdir:str) -> int:
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
    
    def __init__(self, args):
        self.args = args
        self.results_path = os.path.join(os.getcwd(), args["results_path"])
        self.project = args["project"]
        self.num_classes = len(args["names"])
        self.num = first_avaliable(self.results_path, self.project)
        # is resume avaliable
        self.resume = args["resume"] and self.num > 1
        # if need to resume training - load last session
        if self.resume:
            self.num = max(self.num - 1, 1)
        self.project_path = os.path.join(self.results_path, f"{self.project}{self.num}")
        
        self.model: torch.nn.Module = None
    
    def _get_model(self):
        if self.args["model_size"] == "nano":
            return nn.yolo_v11_n(self.num_classes)
        if self.args["model_size"] == "small":
            return nn.yolo_v11_s(self.num_classes)
        if self.args["model_size"] == "medium":
            return nn.yolo_v11_m(self.num_classes)
    
    def _init_model(self):
        # create model
        self.model = self._get_model()
        self.model.cuda()
        
        # load dataset
        self.train_dataset = Dataset(self.args["dataset"], transform=self.train_transform, type="train", mosaic=self.args["mosaic"])
        self.val_dataset = Dataset(self.args["dataset"], transform=self.test_transform, type="val")
        
        # Optimizer
        accumulate = max(round(64 / (self.args["batch_size"])), 1)
        self.args['weight_decay'] *= self.args["batch_size"] * accumulate / 64

        self.optimizer = torch.optim.SGD(util.set_params(model, self.args['weight_decay']),
                                self.args['min_lr'], self.args['momentum'], nesterov=True)

        # EMA
        self.ema = util.EMA(model)

        self.train_loader = data.DataLoader(self.train_dataset, self.args["batch_size"], 
                                num_workers=8, pin_memory=True, collate_fn=Dataset.collate_fn)
        # поменял args["batch_size"] на 1
        self.val_loader = data.DataLoader(self.val_dataset, 1, shuffle=False,
                                num_workers=4, pin_memory=True, collate_fn=Dataset.collate_fn)

        # Scheduler
        num_steps = len(self.train_loader)
        scheduler = util.LinearLR(self.args, num_steps)

        if args["distributed"]:
            # DDP mode
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                            device_ids=[args["local_rank"]],
                                                            output_device=args["local_rank"])

        best = 0
        # amp_scale = torch.amp.GradScaler()
        criterion = util.ComputeLoss(model, params)
        
    
    def _prepare_pransforms(self):
        # TODO: read parameters
        self.train_transform = Compose([
            transforms.MarkScaling(300),        # apply scaling to data in bboxes
            transforms.RandomResize(0.1),       # stretch or compress signal in time domain
            transforms.RandomJitter(0.6),       # add random normal noise
            transforms.TimeMasking(0.2),        # erase random area in time domain
            transforms.RandomCrop(Trainer.INPUT_LENGTH),    # make a random crop to desired size
            transforms.RandomInvert(0.5),       # invert sign of channels
            transforms.RandomReverse(0.5),      # reverse signal in time domain
            transforms.ChannelScaling(),        # apply random scaling to channels
            transforms.ChannelMasking(0.1),     # erase random channels
            ])
        self.test_transform = Compose([
            transforms.Padding(Trainer.REGULAR_LENGTH), # add padding of signal to full size
            transforms.Resize(Trainer.INPUT_LENGTH),    # apply resizing to signal
            ])
        
    def _prepare(self):
        os.makedirs(self.project_path, exist_ok=True)
        # paths to resulting files
        self.logger_path = os.path.join(self.project_path, 'step.csv')
        self.best_path = os.path.join(self.project_path, 'best.pt')
        self.last_path = os.path.join(self.project_path, 'last.pt')
        # logging
        self.log = open(self.logger_path, "w")
        logger = csv.DictWriter(self.logger_path, fieldnames=['epoch',
                                'box_train', 'cls_train', 'dfl_train',
                                'box_val', 'cls_val', 'dfl_val',
                                'Recall', 'Precision', 'mAP@50', 'mAP'])
        if not self.resume:
            logger.writeheader()
        # load state of model
        if self.resume and os.path.exists(self.last_path):
            checkpoint = torch.load(self.last_path, weights_only=True)
            self.model.load_state_dict(checkpoint['model'], strict=False)
            self.start_epoch = checkpoint['epoch']
        
        
        

def train(args, params):
    
    
    

    

    with open(logger_path, 'w') as log:

        for epoch in range(args["epochs"]):
            model.train()
            if args["epochs"] - epoch == 10:
                train_loader.dataset.mosaic = False

            p_bar = enumerate(train_loader)

            if args["local_rank"] == 0:
                print(('\n' + '%10s' * 5) % ('epoch', 'memory', 'box', 'cls', 'dfl'))
                p_bar = tqdm.tqdm(p_bar, total=num_steps)

            optimizer.zero_grad()
            avg_box_loss = util.AverageMeter()
            avg_cls_loss = util.AverageMeter()
            avg_dfl_loss = util.AverageMeter()
            for i, (samples, targets) in p_bar:

                step = i + num_steps * epoch
                scheduler.step(step, optimizer)

                samples = samples.cuda().float() #/ 255

                # Forward
                # with torch.amp.autocast('cuda'):
                outputs = model(samples)  # forward
                loss_box, loss_cls, loss_dfl = criterion(outputs, targets)

                avg_box_loss.update(loss_box.item(), samples.size(0))
                avg_cls_loss.update(loss_cls.item(), samples.size(0))
                avg_dfl_loss.update(loss_dfl.item(), samples.size(0))

                loss_box *= args["batch_size"]  # loss scaled by batch_size
                loss_cls *= args["batch_size"]  # loss scaled by batch_size
                loss_dfl *= args["batch_size"]  # loss scaled by batch_size

                # Backward
                (loss_box + loss_cls + loss_dfl).backward()

                # Optimize
                if step % accumulate == 0:
                    # amp_scale.unscale_(optimizer)  # unscale gradients
                    util.clip_gradients(model)  # clip gradients
                    optimizer.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)

                torch.cuda.synchronize()

                # Log
                if args["local_rank"] == 0:
                    memory = f'{torch.cuda.memory_reserved() / 1E9:.4g}G'  # (GB)
                    s = ('%10s' * 2 + '%10.3g' * 3) % (f'{epoch + 1}/{args["epochs"]}', memory,
                                                       avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg)
                    p_bar.set_description(s)

            if args["local_rank"] == 0:
                # mAP
                last = test(args, params, ema.ema, val_loader)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box_train': str(f'{avg_box_loss.avg:.3f}'),
                                 'cls_train': str(f'{avg_cls_loss.avg:.3f}'),
                                 'dfl_train': str(f'{avg_dfl_loss.avg:.3f}'),
                                 'box_val': str(f'{last[0]:.3f}'),
                                 'cls_val': str(f'{last[1]:.3f}'),
                                 'dfl_val': str(f'{last[2]:.3f}'),
                                 'mAP': str(f'{last[3]:.3f}'),
                                 'mAP@50': str(f'{last[4]:.3f}'),
                                 'Recall': str(f'{last[5]:.3f}'),
                                 'Precision': str(f'{last[6]:.3f}')})
                log.flush()

                # Update best mAP
                if last[0] > best:
                    best = last[0]

                # Save model
                save = {'epoch': epoch + 1,
                        'model': copy.deepcopy(ema.ema)}

                # Save last, best and delete
                torch.save(save, f=last_path)
                if best == last[0]:
                    torch.save(save, f=best_path)
                del save

    if args["local_rank"] == 0:
        util.strip_optimizer(best_path)  # strip optimizers
        util.strip_optimizer(last_path)  # strip optimizers


@torch.no_grad()
def test(args, params, model=None, loader = None):

    plot = False

    model.eval()
    model.set_validate(True)
    # model = model.cuda()
    criterion = util.ComputeLoss(model, params)
    
    # Configure
    iou_v = torch.linspace(start=0.5, end=0.95, steps=10).cuda()  # iou vector for mAP@0.5:0.95
    n_iou = iou_v.numel()

    m_pre = 0
    m_rec = 0
    map50 = 0
    mean_ap = 0
    metrics = []
    p_bar = tqdm.tqdm(loader, desc=('%10s' * 5) % ('', 'precision', 'recall', 'mAP50', 'mAP'))
    # Sum of losses on validation 
    avg_box_loss = util.AverageMeter()
    avg_cls_loss = util.AverageMeter()
    avg_dfl_loss = util.AverageMeter()
    for samples, targets in p_bar:
        samples = samples.cuda()
        # samples = samples.half()  # uint8 to fp16/32 # float16 don't work
        samples = samples.float() #/ 255.  # 0 - 255 to 0.0 - 1.0
        _, _, w = samples.shape  # batch-size, channels, height, width
        scale = torch.tensor((w, w)).cuda()
        # Inference
        raw, outputs = model(samples)
        # Calculate loss
        loss_box, loss_cls, loss_dfl = criterion(raw, targets)
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
    model.set_validate(False)
    # Compute metrics
    metrics = [torch.cat(x, dim=0).cpu().numpy() for x in zip(*metrics)]  # to numpy
    if len(metrics) and metrics[0].any():
        tp, fp, m_pre, m_rec, map50, mean_ap = util.compute_ap(*metrics, plot=plot, names=params["names"])
    # Print results
    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))
    # Return results
    # model.float()  # for training
    return avg_box_loss.avg, avg_cls_loss.avg, avg_dfl_loss.avg, mean_ap, map50, m_rec, m_pre


def profile(args, params):
    import thop
    shape = (1, 30, args["input_size"])
    model = nn.yolo_v11_n(len(params['names'])).fuse()

    model.eval()
    model(torch.zeros(shape))

    x = torch.empty(shape)
    flops, num_params = thop.profile(model, inputs=[x], verbose=False)
    flops, num_params = thop.clever_format(nums=[2 * flops, num_params], format="%.3f")

    if args["local_rank"] == 0:
        print(f'Number of parameters: {num_params}')
        print(f'Number of FLOPs: {flops}')

config = {
    "batch_size": 50,
    "epochs": 150,
    "project": "scnd_try_small",
    "train": True,
    "test": False,
    "dataset": "D:/dataset/data_1d/dataset/dataset.yaml",
    "local_rank": 0,
    "world_size": 1
}

def main():
    args = config

    args["distributed"] = args["world_size"] > 1

    if args["distributed"]:
        torch.cuda.set_device(device=args["local_rank"])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    if args["local_rank"] == 0:
        if not os.path.exists('weights'):
            os.makedirs('weights')

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    util.setup_seed()
    util.setup_multi_processes()

    profile(args, params)

    if args["train"]:
        train(args, params)
    # if args["test"]:
    #     test(args, params, nn.YOLO(num_classes=3))

    # Clean
    if args["distributed"]:
        torch.distributed.destroy_process_group()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    Logger.add_console_handler()
    main()
