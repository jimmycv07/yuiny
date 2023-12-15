import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import os
import csv
import pandas as pd
from dataset import my_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from main import build_model
from config import load_config

'''
learning rate scheduler
tensorboard
'''
device = "cuda:1"

def inference(config):


    # Modify the final classification layer
    num_classes = 4  # Number of classes in your task
    model = build_model(config, num_classes=num_classes)
    model.load_state_dict(torch.load(config.model_weight))
    model.to(device)


    data_dir = '/project/yuiny/test'  # Replace with the path to your dataset
    csv_file = '/project/yuiny/sample_submission.csv'
    test_dataset = my_dataset(config, data_dir, csv_file, split="test")
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

    # Validation
    model.eval()
    predicted_classes = []
    with torch.no_grad():
        with tqdm(test_loader, total=len(test_loader), leave=False) as pbar:
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)  
                # print(inputs, labels)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                predicted_classes.extend(predicted.tolist())
    df = pd.read_csv(csv_file)
    # import ipdb; ipdb.set_trace()
    print(len(predicted_classes), len(df))
    df['Type'] = predicted_classes
    # df['Type'] = predicted
    df.to_csv("logs/efficientnet_b1/yuiny.csv", index=False)

if __name__ == "__main__":
    
    config = load_config("/home/yehhh/yuiny/logs/efficientnet_b1/efficientnet_b1.yaml")
    config.model_weight = "/home/yehhh/yuiny/logs/efficientnet_b1/efficientnet_b1_4_86.pth"
    inference(config)