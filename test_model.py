from nets.nn import *

from utils.dataset import *

from utils import util
from torchinfo import summary
from nets.nn import yolo_v11_l
from test_dataset import get_datasets

import yaml

import matplotlib.pyplot as plt



def show_labels(model, data, targets=None):
    data_np = data.numpy()[0]
    
    # data = data.unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(data)
        outputs = util.non_max_suppression(outputs)
    scale = 6
    plt.plot(data_np.T + scale * numpy.arange(0,data_np.shape[0])[None,:])
    y1 = -2*scale
    y2 = (data_np.shape[0]+2) * scale
    # print(outputs)
    
    for output in outputs:
        for label in output:
            box = label[0:2] / 49152 * data_np.shape[-1]
            x1 = box[0] 
            x2 = box[1]
            
            plt.plot([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1], color = "cyan")
            plt.plot([(x1+x2)/2]*2,[y1,y2], lw=3, color = "cyan")
            plt.text(x1, y1, int(label[3]), color = "cyan", fontsize = 30, va="top")
    
    for cls, box in zip(targets["cls"], targets["box"]):
        # box = label[0:2] / 49152 * data_np.shape[-1]
        x1 = box[0] * data_np.shape[-1]
        x2 = box[1] * data_np.shape[-1]/2
        x1,x2 = x1-x2, x1+x2
        
        plt.plot([x1,x1,x2,x2,x1],[y1,y2,y2,y1,y1], color = "red")
        plt.plot([(x1+x2)/2]*2,[y1,y2], lw=3, color = "red")
        plt.text(x1, y2, int(cls), color = "red", fontsize = 30, va="bottom")
    
    plt.show()

def test_use():
    best_path = os.path.join(os.getcwd(), 'runs/large2/best.pt')
    chkpt = torch.load(f=best_path, map_location='cpu', weights_only=False)
    model = yolo_v11_l(6)
    model.load_state_dict(chkpt['ema']['ema_model_state_dict'])
    model.float().fuse()
    model.eval()
    model.cpu()
    
    d_train, d_test = get_datasets("test")
    
    # filename = r"D:\dataset\data_1d\dataset\data\test\kiwi629_left_300.pt"
    
    # with open(filaname, "rb") as f:
    #     data = torch.load(f, map_location="cpu", weights_only=False)
    
    # data = data.float().cpu()
    # ancors, stride = util.make_anchors(data, model.stride[2:3])
    
    for samples, targets in d_test:
        # print(targets)
        show_labels(model, samples, targets)
    
    
    
    
def test_summary():
    # model = yolo_v11_n(6)
    model = yolo_v11_m(6)
    print(summary(model, input_size=(100,30,49152), depth=3, verbose=2))
    


if __name__ == "__main__":
    test_use()
    # test_summary()
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
    