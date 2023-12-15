import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
import os
import csv
import yaml
import shutil
import random
import argparse
import numpy as np
from dataset import my_dataset
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from config import load_config
from torchvision.transforms import Compose


'''
TODO:
    √ 0.95 more epochs
    0.25 prior
'''

def build_model(config, num_classes=4):
    if config.model == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        fc_0 = nn.Linear(model.fc.in_features, 256)
        fc_1 = nn.Linear(256, num_classes)
        model.fc = nn.Sequential(fc_0, fc_1)
    elif config.model == "efficientnet_b1":
        model = models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
        # for name, m in model.named_modules():
        #     print(name, m)
        # import ipdb; ipdb.set_trace()
    elif config.model == "swin_v2_s":
        model = models.swin_v2_s(weights=models.Swin_V2_S_Weights.DEFAULT)
        # import ipdb; ipdb.set_trace()
        model.head = nn.Linear(in_features=768, out_features=num_classes)

    # if fine_tune:
    #     print('[INFO]: Fine-tuning all layers...')
    #     for params in model.parameters():
    #         params.requires_grad = True
    # elif not fine_tune:
    #     print('[INFO]: Freezing hidden layers...')
    #     for params in model.parameters():
    #         params.requires_grad = False
    # Change the final classification head.
    return model

def train(config):

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    # Freeze all the parameters
    # for param in model.parameters():
    #     param.requires_grad = False

    # # Unfreeze the last 2 or 3 convolutional layers
    # num_layers_to_unfreeze = 2  # Set the number of layers to unfreeze
    # layer_count = 0

    # for child in model.children():
    #     if isinstance(child, torch.nn.Sequential):
    #         for param in child[-num_layers_to_unfreeze:].parameters():
    #             param.requires_grad = True
    #         break
    #     layer_count += 1
    #     if layer_count >= num_layers_to_unfreeze:
    #         for param in child.parameters():
    #             param.requires_grad = True

    # # Verify the requires_grad status of the model's parameters
    # for name, param in model.named_parameters():
    #     print(name, param.requires_grad)

    # Modify the final classification layer
    num_classes = 4  # Number of classes in your task
    
    model = build_model(config, num_classes=num_classes)
    model.to(config.device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=33, gamma=0.1)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-8)
    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-8)

    # Create data loaders (you'll need to define your dataset)
    data_dir = '/project/yuiny/train'  # Replace with the path to your dataset
    csv_file = '/project/yuiny/training_data.csv'
    train_dataset = my_dataset(config, data_dir, csv_file, split="train")

    
    train_indices, val_indices = train_test_split(range(len(train_dataset)), train_size=config.train_val_split, shuffle=True)

    # sampler = SubsetRandomSampler(train_indices)
    # sampler.generator = torch.Generator().manual_seed(config.seed)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler)
    # sampler = SubsetRandomSampler(val_indices)
    # sampler.generator = torch.Generator().manual_seed(config.seed)
    # val_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=sampler)
    
   
    val_sampler = SubsetRandomSampler(val_indices)
    val_sampler.generator = torch.Generator().manual_seed(config.seed)
    val_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=val_sampler, pin_memory=True)

    train_sampler = SubsetRandomSampler(train_indices)
    train_sampler.generator = torch.Generator().manual_seed(config.seed)
    # import ipdb; ipdb.set_trace()
    train_dataset.transform = transforms.Compose([
            train_dataset.transform,
            train_dataset.augmentation
        ])
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, sampler=train_sampler, pin_memory=True)
    # import ipdb; ipdb.set_trace()

 
    # Create a SummaryWriter instance with the specified log directory
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    writer = SummaryWriter(log_dir=config.log_dir)

    # Training loop
    highest_val_accuracy = 0.8
    with tqdm(range(config.num_epochs), desc="Epochs") as epoch_pbar:
        for epoch in epoch_pbar:
            # Training
            model.train()
            train_loss = 0.0
            # with tqdm(train_loader, total=len(train_loader), leave=False, desc=f"Epoch {epoch+1}/{config.num_epochs}") as pbar:
            for inputs, labels in train_loader:
                optimizer.zero_grad()
                # import ipdb; ipdb.set_trace()
                inputs, labels = inputs.to(config.device), labels.to(config.device)   
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()  # Accumulate training loss

            epoch_pbar.set_postfix({"Training Loss": train_loss / (len(train_loader) * config.batch_size)})
            writer.add_scalar('Loss/Train', train_loss / (len(train_loader) * config.batch_size), epoch)

            # if epoch % (config.num_epochs // 10) == 0:
            if 1:
                # Validation
                model.eval()
                val_loss = 0.0
                correct = 0
                total = 0
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(config.device), labels.to(config.device)  
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                val_loss /= (len(val_loader) * config.batch_size)
                val_accuracy = correct / total
                # if val_accuracy > highest_val_accuracy:
                #     highest_val_accuracy = val_accuracy
                torch.save(model.state_dict(), os.path.join(config.log_dir, \
                            f"{config.model}_{epoch}_{int(val_accuracy * 100)}.pth"))

                print(f"Epoch [{epoch+1}/{config.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
                writer.add_scalar('Loss/Val', val_loss , epoch)
                writer.add_scalar('Validation accuracy', val_accuracy , epoch)

        scheduler.step()
    writer.close()

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file")

    args = parser.parse_args()
    config = load_config(args.config)
    
    train(config)