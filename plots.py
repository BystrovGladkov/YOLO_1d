import pandas as pd
import matplotlib.pyplot as plt
import os

if __name__ == "__main__":
    path = "./runs/large2"
    
    filename = os.path.join(path, "step.csv")
    
    if not (os.path.exists(filename) and os.path.isfile(filename)):
        print("File not found")
    
    df = pd.read_csv(filename)
    
    fig, axes = plt.subplots(2,2)
    axes = axes.flatten()
    
    """ losses """
    # train_loss = df["box"] + df["cls"] + df["dfl"]
    train_loss = df["box_train"] + df["cls_train"] + df["dfl_train"]
    val_loss = df["box_val"] + df["cls_val"] + df["dfl_val"]
    
    axes[0].plot(train_loss, label="Train loss")
    axes[0].plot(val_loss, label="VAlidation loss")
    axes[0].legend()
    axes[0].set_title("Loss")
    
    """ precision """
    axes[1].plot(df["Precision"])
    
    axes[1].set_title("Precision")
    
    """ recall """
    axes[2].plot(df["Recall"])
    
    axes[2].set_title("Recall")
    
    
    """ mAP """
    axes[3].plot(df["mAP@50"], label = "mAP@50")
    axes[3].plot(df["mAP"], label = "mAP@50-95")
    
    axes[3].set_title("mAP")
    axes[3].legend()
    
    plt.show()