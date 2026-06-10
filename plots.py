from typing import List
import pandas as pd
import matplotlib.pyplot as plt
import os

plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
root_dir = r"E:\git\neuro\rails\pictures\gladkov_diplom"

def make_plots(filename, name, axes, alpha=1):
    
    if not (os.path.exists(filename) and os.path.isfile(filename)):
        print("File not found")
    
    df = pd.read_csv(filename + "/step.csv")
    
    fig, ax = plt.subplots()
    
    """ losses """
    # train_loss = df["box"] + df["cls"] + df["dfl"]
    train_loss = df["box_train"] + df["cls_train"] + df["dfl_train"]
    val_loss = df["box_val"] + df["cls_val"] + df["dfl_val"]
    
    print("min val loss:", val_loss.min())
    
    ax.plot(train_loss, label="Train loss")
    ax.plot(val_loss, label="Validation loss")
    ax.legend()
    ax.set_ylabel("Loss")
    ax.set_xlabel("Epoch")
    ax.set_title(f"Loss {name}")
    ax.set_ylim(ymin=2, ymax=10)
    fig.set_size_inches(4,4)
    fig.tight_layout()
    fig.savefig(root_dir + f"{name}_loss.png",dpi=300)
    
    """ precision """
    axes[0].plot(df["Precision"], label=f"{name}", alpha=alpha)
    
    """ recall """
    axes[1].plot(df["Recall"], label=f"{name}", alpha=alpha)
    
    """ mAP """
    axes[2].plot(df["mAP@50"], label = f"{name}", alpha=alpha)
    axes[3].plot(df["mAP"], label = f"{name}", alpha=alpha)
    
    

if __name__ == "__main__":
    # path = "./runs/test_not_change_arch1"
    paths = ["./runs/v0.0_2", "./runs/v0.1_1", "./runs/v0.2_1", "./runs/v0.4_1", "./runs/v0.5_1"]
    paths = ["./runs/v0.0_large1", "./runs/v0.1_large1", "./runs/v0.2_large1", "./runs/v0.3_large1", "./runs/v0.4_large1", "./runs/v0.5_large1", "./runs/v0.5_large_no_occlusion1"]
    
    fig, axes = plt.subplots(2,2)
    axes :List[plt.Axes]= axes.flatten()
    
    for p in paths:
        name = p.rsplit("/", maxsplit=1)[-1].split("_")[0]
        make_plots(p, name, axes, alpha=1)
        
    axes[0].set_title("Precision")
    axes[1].set_title("Recall")
    axes[2].set_title("mAP@50")
    axes[3].set_title("mAP@50-95")
    
    axes[2].set_xlabel("Epoch")
    axes[3].set_xlabel("Epoch")
    
    axes[0].set_ylim(0.75, 1)
    axes[1].set_ylim(0.75, 1)
    axes[2].set_ylim(0.75, 1)
    axes[3].set_ylim(0.65, 0.9)
    fig.set_size_inches(10, 5)
    axes[1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5)
    )
    fig.tight_layout()
    # fig.subplots_adjust(right=0.85)
    fig.savefig(root_dir + "yolo1d_validation.png")
    
    
    
    plt.show()