from dataset import CT2US
from models.unet import UNet
from engine import trainer
from utils import plot_results

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import random

import yaml
import json
import time
import os


config = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)

LEARNING_RATE = float(config["LEARNING_RATE"])
BATCH_SIZE = int(config["BATCH_SIZE"])
NUM_EPOCHS = int(config["NUM_EPOCHS"])

IMAGE_SIZE = int(config["IMAGE_SIZE"])
THRESHOLD = float(config["THRESHOLD"])
MODEL = config["MODEL"]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using {DEVICE} device")


def START_seed():
    seed = 9
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    START_seed()

    #run id is date and time of the run
    run_id = time.strftime("%Y-%m-%d_%H-%M-%S")

    #create folder for this run in runs folder
    os.mkdir("./runs/" + run_id)

    save_dir = "./runs/" + run_id
    

    #load data
    image_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    mask_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
    ])

    dataset = CT2US(root="datasets/CT2US", image_transform=image_transform, mask_transform=mask_transform)

    train_set, test_set = train_test_split(dataset, test_size=0.2, random_state=42)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

    #load model
    if MODEL == "UNet":
        model = UNet(outSize=(IMAGE_SIZE, IMAGE_SIZE)).to(DEVICE)
    else:
        raise Exception("Model not implemented")
    
    #load optimizer
    loss = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)


    #train model
    results = trainer(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        loss_fn=loss,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=DEVICE,
        epochs=NUM_EPOCHS,
        save_dir=save_dir,
    )

    train_summary = {
        "config": config,
        "results": results,
    }

    with open(save_dir + "/train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=4)

    plot_results(results, save_dir)




if __name__ == "__main__":
    main()


