import os
import csv
import torch
from PIL import Image
import torch.utils.data as Data
import torchvision.transforms as transforms


class MyDataset(Data.Dataset):  # 需要继承data.Dataset
    def __init__(self, transform=transforms.ToTensor()):
        # TODO
        # 1. Initialize file path or list of file names.
        self.root = './data/'
        self.transform = transform
        self.files = []
        self.label = []
        with open('./test_data.csv', 'r') as f:
            csv_reader = csv.reader(f)
            for row in csv_reader:
                self.files.append([row[0], row[1], row[2], row[3]])
                self.label.append(int(row[4]))

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        imgs = []
        for i in range(4):
            img = None
            file_path = self.root + self.files[index][i]
            if os.path.exists(file_path):
                try:
                    img = Image.open(file_path).convert('RGB').resize((448, 448))
                    img = self.transform(img)
                except:
                    print("图片无法打开 ", file_path)
            if img is None:
                print(file_path)
                print("未找到图片: ", self.files[index], self.label[index])
            else:
                imgs.append(img)
        if len(imgs) < 4:
            print("图片不全: ", self.files[index], self.label[index])
        imgs = torch.cat(imgs)
        target = float(self.label[index])
        return imgs.view(1, 4, 3, 448, 448), target

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.files)

