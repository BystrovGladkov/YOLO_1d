from utils.dataset import Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms.v2 import Compose
import utils.transforms as transforms

if __name__ == "__main__":
    
    # length on one signal sample
    REGULAR_LENGTH = 50000
    # length of network input
    INPUT_LENGTH = 49152
    
    
    train_transform = Compose(
        [transforms.RandomResize(0.1),
        transforms.RandomJitter(0.2),
        transforms.TimeMasking(0.2),
        transforms.RandomCrop(INPUT_LENGTH),
        transforms.RandomInvert(0.5),
        transforms.RandomReverse(0.5),
        transforms.ChannelMasking(0.1),]
    )
    test_transform = Compose(
        [transforms.Padding(REGULAR_LENGTH),
        transforms.Resize(INPUT_LENGTH),]
    )
    
    d_train = Dataset("D:\dataset\data_1d\dataset\dataset.yaml", params=None, 
                transform=train_transform, type="val")
    
    d_test = Dataset("D:\dataset\data_1d\dataset\dataset.yaml", params=None, 
                transform=test_transform, type="val")
    
    # print(d.filenames)
    
    scale = 10
    y1,y2 = -scale*2, scale*32
    
    a = iter(d_train)
    b = iter(d_test)
    while(a and b):
        fig, axes = plt.subplots(2,1)
        for index, c in enumerate([a,b]):
            train_d = next(c)
            data = train_d[0]
            
            axes[index].plot(data.numpy().T + np.arange(data.shape[0])[None,:]*scale)
            box = train_d[2]
            x1 = (box[:,0] - box[:,1]/2)*INPUT_LENGTH
            x2 = (box[:,0] + box[:,1]/2)*INPUT_LENGTH
            
            for l,r in zip(x1,x2):
                axes[index].plot([x1,x2,x2,x1,x1], [y1,y1,y2,y2,y1])
        
        plt.show()