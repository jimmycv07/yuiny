import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import RandomHorizontalFlip, RandomRotation



class my_dataset(Dataset):
    def __init__(self, config, data_dir, csv_file, split="train"):
        self.data_dir = data_dir
        self.file_list = []
        self.labels = []
        self.split = split
        with open(csv_file, 'r') as file:
            csv_reader = csv.reader(file)
            next(csv_reader)  # Skip the header row
            for row in csv_reader:
                file_name, label = row
                self.file_list.append(file_name)
                self.labels.append(int(label))
        # import ipdb; ipdb.set_trace()
        # if self.split == "train":
        if config.model == "resnet50":
            resize = transforms.Resize((224, 224))
        elif config.model == "efficientnet_b1":
            resize = transforms.Resize((240, 240))
        elif config.model == "swin_v2_s":
            resize = transforms.Compose([ \
                transforms.Resize(260, interpolation=Image.BICUBIC), \
                transforms.CenterCrop(256)
            ])
            # import ipdb; ipdb.set_trace()
        self.augmentation = transforms.Compose([
            RandomHorizontalFlip(),  # Randomly flip the image horizontally with a probability of 0.5
            RandomRotation(degrees=15)    # Randomly rotate the image by a maximum of 15 degrees
        ])
        self.transform = transforms.Compose([
            resize,
            transforms.ToTensor(),           
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize image values
        ])
        


    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.file_list[idx])
        image = Image.open(img_name)
        # :q check the chanel of image
        num_channels = len(image.getbands())
        if num_channels == 1:
            image = image.convert("RGB")
        # print(len(image.getbands()), idx)
    
        image = self.transform(image)
        # import ipdb; ipdb.set_trace()
        class_label = self.labels[idx]
        # import ipdb; ipdb.set_trace()
        return image, class_label

if __name__ == "__main__":
    # Create an instance of the custom dataset
    data_dir = '/project/yuiny/train'  # Replace with the path to your dataset
    data_dir = '/project/yuiny/test'  # Replace with the path to your dataset
    csv_file = '/project/yuiny/training_data.csv'
    csv_file = '/project/yuiny/sample_submission.csv'
    # custom_dataset = my_dataset(data_dir, csv_file)
    custom_dataset = my_dataset(data_dir, csv_file, split="test")

    # Create a data loader for the custom dataset
    batch_size = 128  # Adjust as needed
    data_loader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    # Iterate through the dataset using the data loader
    for images, labels in data_loader:
        # Perform operations on each batch of data (e.g., training)
        pass

