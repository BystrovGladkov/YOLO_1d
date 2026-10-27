import copy
import random
from time import time

import math
import numpy
import torch
import torchvision
from torch.nn.functional import cross_entropy


def setup_seed(seed: int = 0):
    """
    Setup random seed.
    """
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_multi_processes():
    """
    Setup multi-processing environment variables.
    """
    import cv2
    from os import environ
    from platform import system

    # set multiprocess start method as `fork` to speed up the training
    if system() != 'Windows':
        torch.multiprocessing.set_start_method('fork', force=True)

    # disable opencv multithreading to avoid system being overloaded
    cv2.setNumThreads(0)

    # setup OMP threads
    if 'OMP_NUM_THREADS' not in environ:
        environ['OMP_NUM_THREADS'] = '1'

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in environ:
        environ['MKL_NUM_THREADS'] = '1'


def export_onnx(args):
    import onnx  # noqa

    inputs = ['images']
    outputs = ['outputs']
    dynamic = {'outputs': {0: 'batch', 1: 'anchors'}}

    m = torch.load('./weights/best.pt')['model'].float()
    x = torch.zeros((1, 15, args.input_size))

    torch.onnx.export(m.cpu(), x.cpu(),
                      f='./weights/best.onnx',
                      verbose=False,
                      opset_version=12,
                      # WARNING: DNN inference with torch>=1.12 may require do_constant_folding=False
                      do_constant_folding=True,
                      input_names=inputs,
                      output_names=outputs,
                      dynamic_axes=dynamic or None)

    # Checks
    model_onnx = onnx.load('./weights/best.onnx')  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model

    onnx.save(model_onnx, './weights/best.onnx')
    # Inference example
    # https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/autobackend.py


def wh2xy(x):
    y = x.clone() if isinstance(x, torch.Tensor) else numpy.copy(x)
    y[:, 0] = x[:, 0] - x[:, 1] / 2  # top left x
    y[:, 1] = x[:, 0] + x[:, 1] / 2  # top left y
    return y


def make_anchors(x, strides, offset=0.5):
    assert x is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = x[0].dtype, x[0].device
    for i, stride in enumerate(strides):
        _, _, w = x[i].shape
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset  # shift x
        anchor_tensor.append(sx.view(-1,1))
        stride_tensor.append(torch.full((w,1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_tensor), torch.cat(stride_tensor)

def compute_metric(output, target, iou_v):
    # intersection(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2) = target[:, 1:].unsqueeze(1).chunk(2,2)
    (b1, b2) = output[:, :2].unsqueeze(0).chunk(2,2)
    intersection = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)
    # IoU = intersection / (area1 + area2 - intersection)
    iou = intersection / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - intersection + 1e-7)

    correct = numpy.zeros((output.shape[0], iou_v.shape[0]))
    correct = correct.astype(bool)
    for i in range(len(iou_v)):
        # IoU > threshold and classes match
        x = torch.where((iou >= iou_v[i]) & (target[:, 0:1] == output[:, 3]))
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1),
                                 iou[x[0], x[1]][:, None]), 1).cpu().numpy()  # [label, detect, iou]
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[numpy.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[numpy.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.tensor(correct, dtype=torch.bool, device=output.device)

def suppress_contained(output, main_classes, secondary_classes, threshold=0.8):
    """Постобработка после NMS: удаление вторичных объектов, вложенных в главный.

    Объект вторичного класса удаляется, если доля его длины внутри объекта главного класса больше
    `threshold`, а уверенность главного объекта не ниже его собственной. Обход — по убыванию уверенности,
    как в NMS: удалённый объект сам ничего не удаляет; при равной уверенности старшим считается тот,
    что раньше в выдаче. Если класс и главный, и вторичный (например, стрелочный перевод), из двух
    вложенных объектов остаётся более уверенный — так убираются дубли, которые NMS по IoU пропускает
    (у вложенного короткого объекта IoU с длинным мал).

    output: (n, ≥4) [x1, x2, conf, cls, ...] — детекции одного сигнала; возвращается подмножество строк
    в исходном порядке.
    """
    n = output.shape[0]
    if n == 0 or not main_classes or not secondary_classes:
        return output
    order = torch.argsort(output[:, 2], descending=True, stable=True)
    det = output[order]
    cls = det[:, 3].long()
    is_main = torch.isin(cls, torch.as_tensor(list(main_classes), device=det.device))
    is_secondary = torch.isin(cls, torch.as_tensor(list(secondary_classes), device=det.device))
    if not is_main.any() or not is_secondary.any():
        return output
    x1, x2 = det[:, 0], det[:, 1]
    inter = (torch.min(x2[:, None], x2[None, :]) - torch.max(x1[:, None], x1[None, :])).clamp(min=0)
    inside = inter / (x2 - x1).clamp(min=1e-7)[:, None]        # inside[j, i] — доля j внутри i
    keep = torch.ones(n, dtype=torch.bool, device=det.device)
    for j in range(1, n):
        if is_secondary[j]:
            senior = keep[:j] & is_main[:j]
            if (inside[j, :j][senior] > threshold).any():
                keep[j] = False
    keep_original = torch.zeros(n, dtype=torch.bool, device=det.device)
    keep_original[order] = keep
    return output[keep_original]


def non_max_suppression(outputs, confidence_threshold=0.001, iou_threshold=0.65):
    max_wh = 50000 # 
    max_det = 300 # Максимальное количество предсказанных bbox
    max_nms = 30000 # Количество bbox, рассматриваемх как кандидаты на ответ

    bs = outputs.shape[0]  # batch size
    nc = outputs.shape[1] - 2  # number of classes
    xc = outputs[:, 2:2 + nc].amax(1) > confidence_threshold  # candidates

    # Settings
    start = time()
    limit = 0.5 + 0.05 * bs  # seconds to quit after
    output = [torch.zeros((0, 4), device=outputs.device)] * bs
    for index, x in enumerate(outputs):  # image index, image inference
        x = x.transpose(0, -1)[xc[index]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # matrix nx4 (box, confidence, cls)
        box, cls = x.split((2, nc), 1)
        box = wh2xy(box)  # (cx, w) to (x1, x2)
        if nc > 1:
            i, j = (cls > confidence_threshold).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 2 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > confidence_threshold]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        x = x[x[:, 2].argsort(descending=True)[:max_nms]]  # sort by confidence and remove excess boxes

        # Batched NMS
        c = x[:, 3:4] * max_wh  # classes
        boxes, scores = x[:, :2] + c, x[:, 2]  # boxes, scores
        # Далее хочу из 1d bbox сделать 2d bbox, для этого сделаю всем одинаковый низ и верх
        nboxes = boxes.shape[0]
        y1 = torch.zeros((nboxes, 1)).to(boxes.device)  # [N, 1]
        y2 = torch.ones((nboxes, 1)).to(boxes.device)   # [N, 1]
        # Объединяем в [x1, y1, x2, y2]
        boxes = torch.cat([boxes[:, 0:1], y1, boxes[:, 1:2], y2], dim=1)

        indices = torchvision.ops.nms(boxes, scores, iou_threshold)  # NMS
        indices = indices[:max_det]  # limit detections

        output[index] = x[indices]
        if (time() - start) > limit:
            break  # time limit exceeded

    return output


def smooth(y, f=0.1):
    # Box filter of fraction f
    nf = round(len(y) * f * 2) // 2 + 1  # number of filter elements (must be odd)
    p = numpy.ones(nf // 2)  # ones padding
    yp = numpy.concatenate((p * y[0], y, p * y[-1]), 0)  # y padded
    return numpy.convolve(yp, numpy.ones(nf) / nf, mode='valid')  # y-smoothed


def plot_pr_curve(px, py, ap, names, save_dir):
    from matplotlib import pyplot
    fig, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = numpy.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}")  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color="grey")  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color="blue", label="all classes %.3f mAP@0.5" % ap[:, 0].mean())
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title("Precision-Recall Curve")
    fig.savefig(save_dir, dpi=250)
    pyplot.close(fig)


def plot_curve(px, py, names, save_dir, x_label="Confidence", y_label="Metric"):
    from matplotlib import pyplot

    figure, ax = pyplot.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f"{names[i]}")  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")  # plot(confidence, metric)

    y = smooth(py.mean(0), f=0.05)
    ax.plot(px, y, linewidth=3, color="blue", label=f"all classes {y.max():.3f} at {px[y.argmax()]:.3f}")
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title(f"{y_label}-Confidence Curve")
    figure.savefig(save_dir, dpi=250)
    pyplot.close(figure)


def compute_ap(tp, conf, output, target, plot=False, eps=1E-16):
    """
    Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Object-ness value from 0-1 (nparray).
        output:  Predicted object classes (nparray).
        target:  True object classes (nparray).
    # Returns
        The average precision
    """
    # Sort by object-ness
    i = numpy.argsort(-conf)
    tp, conf, output = tp[i], conf[i], output[i]

    # Find unique classes
    unique_classes, nt = numpy.unique(target, return_counts=True)
    nc = unique_classes.shape[0]  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    p = numpy.zeros((nc, 1000))
    r = numpy.zeros((nc, 1000))
    ap = numpy.zeros((nc, tp.shape[1]))
    px, py = numpy.linspace(start=0, stop=1, num=1000), []  # for plotting
    for ci, c in enumerate(unique_classes):
        i = output == c
        nl = nt[ci]  # number of labels
        no = i.sum()  # number of outputs
        if no == 0 or nl == 0:
            continue

        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum(0)
        tpc = tp[i].cumsum(0)

        # Recall
        recall = tpc / (nl + eps)  # recall curve
        # negative x, xp because xp decreases
        r[ci] = numpy.interp(-px, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)  # precision curve
        p[ci] = numpy.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            m_rec = numpy.concatenate(([0.0], recall[:, j], [1.0]))
            m_pre = numpy.concatenate(([1.0], precision[:, j], [0.0]))

            # Compute the precision envelope
            m_pre = numpy.flip(numpy.maximum.accumulate(numpy.flip(m_pre)))

            # Integrate area under curve
            x = numpy.linspace(start=0, stop=1, num=101)  # 101-point interp (COCO)
            ap[ci, j] = numpy.trapz(numpy.interp(x, m_rec, m_pre), x)  # integrate
            if plot and j == 0:
                py.append(numpy.interp(px, m_rec, m_pre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + eps)
    # if plot:
    #     names = dict(enumerate(names))  # to dict
    #     names = [v for k, v in names.items() if k in unique_classes]  # list: only classes that have data
    #     plot_pr_curve(px, py, ap, names, save_dir="./weights/PR_curve.png")
    #     plot_curve(px, f1, names, save_dir="./weights/F1_curve.png", y_label="F1")
    #     plot_curve(px, p, names, save_dir="./weights/P_curve.png", y_label="Precision")
    #     plot_curve(px, r, names, save_dir="./weights/R_curve.png", y_label="Recall")
    i = smooth(f1.mean(0), 0.1).argmax()  # max F1 index
    p, r, f1 = p[:, i], r[:, i], f1[:, i]
    tp = (r * nt).round()  # true positives
    fp = (tp / (p + eps) - tp).round()  # false positives
    ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
    # m_pre, m_rec = p.mean(), r.mean()
    # map50, mean_ap = ap50.mean(), ap.mean()
    return tp, fp, p, r, ap50, ap


def compute_iou(box1, box2, eps=1e-7):
    # Returns Intersection over Union (IoU) of box1(1,2) to box2(n,2)

    # Get the coordinates of bounding boxes
    b1_x1, b1_x2 = box1.chunk(2, -1)
    b2_x1, b2_x2 = box2.chunk(2, -1)
    w1 = b1_x2 - b1_x1 + eps
    w2 = b2_x2 - b2_x1 + eps

    # Intersection area
    inter = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0)

    # Union Area
    union = w1 + w2 - inter + eps

    # IoU
    iou = inter / union
    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)  # convex (smallest enclosing box) width
    c2 = cw ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2) / 4  # center dist ** 2
    # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
    return iou - (rho2 / c2)  # CIoU

def clip_gradients(model, max_norm=10.0):
    parameters = model.parameters()
    torch.nn.utils.clip_grad_norm_(parameters, max_norm=max_norm)


def load_weight(model, ckpt):
    dst = model.state_dict()
    src = torch.load(ckpt)['model'].float().cpu()

    ckpt = {}
    for k, v in src.state_dict().items():
        if k in dst and v.shape == dst[k].shape:
            ckpt[k] = v

    model.load_state_dict(state_dict=ckpt, strict=False)
    return model


def set_params(model, decay):
    """
        Sets a regularization for parameters. 
        For bias and BatchNorm regularization is not used.
    """
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if not p.requires_grad:
                continue
            if n == "bias":  # bias (no decay)
                p1.append(p)
            elif n == "weight" and isinstance(m, norm):  # norm-weight (no decay)
                p1.append(p)
            else:
                p2.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


def plot_lr(args, optimizer, scheduler, num_steps):
    from matplotlib import pyplot

    optimizer = copy.copy(optimizer)
    scheduler = copy.copy(scheduler)

    y = []
    for epoch in range(args.epochs):
        for i in range(num_steps):
            step = i + num_steps * epoch
            scheduler.step(step, optimizer)
            y.append(optimizer.param_groups[0]['lr'])
    pyplot.plot(y, '.-', label='LR')
    pyplot.xlabel('step')
    pyplot.ylabel('LR')
    pyplot.grid()
    pyplot.xlim(0, args.epochs * num_steps)
    pyplot.ylim(0)
    pyplot.savefig('./weights/lr.png', dpi=200)
    pyplot.close()

from yolo_1d.utils.dataset import Dataset

def get_data_weights(dataset: Dataset, eps=1e-5):
    """ Calculates a weights of data samples to fix unbalanced classes. """
    data_to_class = []
    class_counts = numpy.zeros(dataset.num_classes, dtype=numpy.float32)
    for labels in dataset.labels.values():
        data_to_class.append(labels[:, 0].astype(int))
        for i in labels[:, 0]:
            class_counts[int(i)] += 1
            
    inv_freq = 1.0 / (class_counts + eps)  
    weights = []
    for classes in data_to_class:
        if len(classes) == 0:
            weights.append(1.0)
        else:
            w = float(inv_freq[classes].max())  # или .mean()
            weights.append(w)
    weights = numpy.array(weights, dtype=numpy.float32)
    # нормируем, чтобы средний вес ~1
    weights = weights / (weights.mean() + eps)
    return weights

class CosineLR:
    def __init__(self, args, params, num_steps):
        max_lr = params['max_lr']
        min_lr = params['min_lr']

        warmup_steps = int(max(params['warmup_epochs'] * num_steps, 100))
        decay_steps = int(args.epochs * num_steps - warmup_steps)

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps))

        decay_lr = []
        for step in range(1, decay_steps + 1):
            alpha = math.cos(math.pi * step / decay_steps)
            decay_lr.append(min_lr + 0.5 * (max_lr - min_lr) * (1 + alpha))

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class LinearLR:
    def __init__(self, min_lr:float, max_lr:float, warmup_epochs:int, epochs:int, num_steps:int):
        total_steps = int(epochs * num_steps)
        # прогрев не длиннее всего обучения (иначе при коротких прогонах число шагов затухания < 0)
        warmup_steps = min(int(max(warmup_epochs * num_steps, 100)), total_steps)
        decay_steps = total_steps - warmup_steps

        warmup_lr = numpy.linspace(min_lr, max_lr, int(warmup_steps), endpoint=False)
        decay_lr = numpy.linspace(max_lr, min_lr, decay_steps)

        self.total_lr = numpy.concatenate((warmup_lr, decay_lr))

    def step(self, step, optimizer):
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.total_lr[step]


class EMA:
    """
    Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    """

    def __init__(self, model:torch.nn.Module, decay:float=0.9999, tau:int=2000, updates:int=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        # decay exponential ramp (to help early epochs)
        self.tau = tau
        self.decay = decay
        self.weight_decay = lambda x: self.decay * (1 - math.exp(-x / self.tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, model:torch.nn.Module):
        if hasattr(model, 'module'):
            model = model.module
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.weight_decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def state_dict(self) -> dict:
        return {
            'updates': self.updates,
            'decay': self.decay,
            'tau': self.tau,
            'ema_model_state_dict': self.ema.state_dict()
        }

    def load_state_dict(self, state_dict: dict):
        self.updates = state_dict['updates']
        self.decay = state_dict['decay']
        self.tau = state_dict['tau']
        self.weight_decay = lambda x: self.decay * (1 - math.exp(-x / self.tau))
        self.ema.load_state_dict(state_dict['ema_model_state_dict'])
    

class AverageMeter:
    def __init__(self):
        self.num = 0
        self.sum = 0
        self.avg = 0

    def update(self, v, n):
        if not math.isnan(float(v)):
            self.num = self.num + n
            self.sum = self.sum + v * n
            self.avg = self.sum / self.num

def get_adversal_examples(model:torch.nn.Module, criterion, grad_scaler:torch.amp.GradScaler,
                          images: torch.Tensor, targets: torch.Tensor):
    
    # fraction of signals power to change
    abs_scale = 0.3
    power_scale = 0.1
    # add gradient to images
    images = images.clone().detach().requires_grad_(True)
    
    # turn off gradients
    prev_requires = []
    for p in model.parameters():
        prev_requires.append(p.requires_grad)
        p.requires_grad_(False)
    # compute loss
    with torch.amp.autocast('cuda'):
        outputs = model(images)
        loss_box, loss_cls, loss_dfl = criterion(outputs, targets)
    
    loss = (loss_box + loss_cls + loss_dfl)
    # calculate gradient
    grad_x = torch.autograd.grad(grad_scaler.scale(loss), images, retain_graph=False, create_graph=False)[0]
    # calculate powers of signal and gradient
    power = images.square().sum(dim=-1) / images.size(-1)
    max_abs = images.abs().max(dim=-1).values
    grad_power = grad_x.square().sum(dim=-1) / grad_x.size(-1)
    grad_max_abs = grad_x.abs().max(dim=-1).values
    # calculate a scale
    scale = torch.min(power_scale * power / grad_power, abs_scale * max_abs / grad_max_abs)
    # add gradient to images
    # images = images + grad_x.sign() * images * scale # * scale[..., None]
    images = images + grad_x * scale[..., None]
    
    # turn on_gradients
    for p, r in zip(model.parameters(), prev_requires):
        p.requires_grad_(r)
        
    return images.detach()