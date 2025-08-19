from nets.nn import *

from utils.dataset import *

from utils import util
# from torchsummary import summary
from torchinfo import summary

import yaml

import matplotlib.pyplot as plt

def test_use():
    best_path = os.path.join(os.getcwd(), 'fst_try/best.pt')
    model = torch.load(f=best_path, map_location='cpu')
    model = model['model'].float().fuse()
    model.eval()
    model.cpu()
    print(model.stride)
    
    
    data_np = numpy.frombuffer(open(r"D:\dataset\data_1d\data\val\artc352_right_179.bin","rb").read(),dtype=numpy.float32).reshape(15, -1)
    # data_np = numpy.frombuffer(open(r"D:\dataset\data_1d\data\train\kiwi629_left_7.bin","rb").read(),dtype=numpy.float32).reshape(15, -1)
    
    
    
    
    data = torch.from_numpy(data_np.copy())
    data = data.float().cpu()
    # ancors, stride = util.make_anchors(data, model.stride[2:3])
    
    # print(ancors)
    # print(stride)
    
    data = data.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(data)
        outputs = util.non_max_suppression(outputs)
    scale = 6
    plt.plot(data_np.T + scale * numpy.arange(0,15)[None,:])
    y1 = -scale
    y2 = 16 * scale
    print(outputs)
    
    for i, output in enumerate(outputs):
        # idx = targets['idx'] == i
        # cls = output[3]
        for label in output:
            box = label[0:2] / 49152 * data_np.shape[-1]
            x1 = box[0] 
            x2 = box[1]
            print(box)
            plt.plot([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1])
    plt.show()
    
def test_summary():
    model = yolo_v11_n(6)
    print(summary(model, input_size=(100,15,49152)))
    
    

if __name__ == "__main__":
    # test_use()
    test_summary()
if False:
    
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
    