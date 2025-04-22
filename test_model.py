from nets.nn import *

from utils.dataset import *

from torchsummary import summary

import yaml

if __name__ == "__main__":
    # d = yaml.safe_load(open(r"D:\dataset\flat_5level.yaml","r",encoding="utf-8"))
    
    ata = numpy.random.normal((15,50000))
    
    a, t = resize(ata, 49152,augment=False)
    labels = [[0.2,0.1]]
    
    for label in labels:
        label[0]
    
    
    
    
    
    device = torch.device("cuda", 0)
    model = YOLO(num_classes=4)
    # with torch.no_grad():
    summary(model,input_size=(15,49152), batch_size=20, device="cpu")
    
    tensor = torch.rand((1, 15, 49152),dtype=torch.float32,device=device)
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        
        result = model(tensor)
        print(result.size())
    