from yolo_1d.utils.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.v2 import Compose
import yolo_1d.utils.transforms as transforms
import yaml
import torch

from yolo_1d.nets.nn import YOLO, yolo_v11_l
from yolo_1d.nets.criterion import ComputeLoss

# length on one signal sample
REGULAR_LENGTH = 50000
# length of network input
INPUT_LENGTH = 49152


def get_datasets(type = "val"):
    train_transform = Compose([
        transforms.MarkScaling(300),    
        transforms.RandomResize(0.1),
        transforms.PhaseSpectrumNoise(0.5), # adds noise in phase spectrum
        transforms.RandomJitter(0.6),
        transforms.TimeMasking(0.2),
        transforms.RandomCrop(INPUT_LENGTH),
        transforms.RandomInvert(0.5),
        transforms.RandomReverse(0.5),
        transforms.ChannelScaling(),
        transforms.ChannelMasking(0.1),
        ])
    # train_transform = Compose([
    #     transforms.MarkScaling(300),    
    #     transforms.Padding(REGULAR_LENGTH),
    #     transforms.Resize(INPUT_LENGTH),
    #     ])
    
    test_transform = Compose(
        [transforms.Padding(REGULAR_LENGTH),
        transforms.Resize(INPUT_LENGTH),]
    )
    
    d_train = Dataset("D:\dataset\data_1d\dataset\dataset.yaml", 
                transform=train_transform, type=type, mosaic_prob=0.0)
    
    d_test = Dataset("D:\dataset\data_1d\dataset\dataset.yaml", 
                transform=test_transform, type=type, mosaic_prob=0.0)
    
    d_train = torch.utils.data.DataLoader(d_train, 1, shuffle=False, collate_fn=d_train.collate_fn)
    d_test = torch.utils.data.DataLoader(d_test, 1, shuffle=False, collate_fn=d_train.collate_fn)
    
    return d_train, d_test
    
    
def test_adversals():
    
    model = yolo_v11_l(6)
    
    check_path = "runs/large1/last.pt"
    
    
    checkpoint = torch.load(check_path, weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    criterion = ComputeLoss(model, 7.5, 0.5, 1.1)
    
    d_train, d_test = get_datasets()
    
    
    scale = 10
    y1,y2 = -scale*2, scale*32
    for a, b in zip(d_train, d_test):
    # while(d_train and d_test):
        fig, axes = plt.subplots(2,1)
        for index, c in enumerate([a, b]):
            # train_d = next(c)
            
            if index == 0:
                data = get_adversals(model, criterion, c[0], c[1]).squeeze(0)
            else:
                data = c[0].squeeze(0)
            
            axes[index].plot(data.numpy().T + np.arange(data.shape[0])[None,:]*scale)
            
            box = c[1]["box"]
            x1 = (box[:,0] - box[:,1]/2) * INPUT_LENGTH
            x2 = (box[:,0] + box[:,1]/2) * INPUT_LENGTH
            
            # for l, r in zip(x1,x2):
            axes[index].plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1])
        
        plt.show()
    

if __name__ == "__main__":
    
    
    
    # test_adversals()
    
    # print(d.filenames)
    
    scale = 10
    y1,y2 = -scale*2, scale*32
    
    d_train, d_test = get_datasets()
    
    n = 10
    i = 0
    
    a = iter(d_train)
    b = iter(d_test)
    while(a and b):
        i += 1
        if i >= n:
            fig, axes = plt.subplots(2,1)
            print(i)
        for index, c in enumerate([a,b]):
            train_d = next(c)
            if i < n:
                continue
            data = train_d[0].squeeze_(0)
            
            axes[index].plot(data.numpy().T + np.arange(data.shape[0])[None,:]*scale)
            box = train_d[1]["box"]
            x1 = (box[:,0] - box[:,1] / 2) * INPUT_LENGTH
            x2 = (box[:,0] + box[:,1] / 2) * INPUT_LENGTH
            
            for l,r in zip(x1,x2):
                axes[index].plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1])
        if i >= n:
            plt.show()