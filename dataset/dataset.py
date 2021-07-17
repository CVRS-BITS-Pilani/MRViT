import torch
import numpy as np
import torch.utils.data as data
import h5py
from medmnist.info import INFO
import torchvision.transforms as transforms

data_flag = 'retinamnist'

info = INFO[data_flag]
task = info['task']

class MyDataset(data.Dataset):
    """dataset class for returning both small and large images along with labels."""
    def __init__(self, images_small, images_large, labels, transform=None):
        self.images_small = images_small
        self.images_large = images_large
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_s = self.images_small[index].permute(2, 1, 0)
        image_l = self.images_large[index].permute(2, 1, 0)
        label = torch.as_tensor(self.labels[index]).to(torch.int64)

        if self.transform is not None:
            image_s = self.transform(image_s)
            image_l = self.transform(image_l)

        return image_s, image_l , label

    def get_images(self):
        images_s = self.images_small.permute(0, 3, 2, 1)
        images_l = self.images_large
        images_l = images_l.permute(0, 3, 2, 1)
        if self.transform is not None:
            for image in images_s[:]:
                image = self.transform(image)
            for image in images_l[:]:
                image = self.transform(image)
        return images_s, images_l

    def get_labels(self): 
        return self.labels.flatten()

def get_datasets():
    """retrieving train, validation and test datasets."""
    with h5py.File(f'/content/drive/MyDrive/MRViT/data/{data_flag}_super.h5', 'r') as hf:
        print(hf.keys())
        trainl = torch.from_numpy(np.array(hf.get('train'), dtype= 'float32'))
        testl_data = torch.from_numpy(np.array(hf.get('test'), dtype= 'float32'))
        trains = torch.from_numpy(np.array(hf.get('train_small'), dtype= 'float32'))
        tests_data = torch.from_numpy(np.array(hf.get('test_small'), dtype= 'float32'))
        train_labelo = torch.from_numpy(np.array(hf.get('label_train'), dtype= 'float32'))
        test_label = torch.from_numpy(np.array(hf.get('label_test'), dtype= 'float32'))
        train_labelo = train_labelo.type(torch.float32)
        test_label = test_label.type(torch.float32)
        print(testl_data.shape, tests_data.shape, test_label.shape)

    train_size = len(trainl)
    val_size = int(0.05 * train_size)

    rand_ind = np.random.randint(0, train_size, train_size)

    train_ind = rand_ind[val_size:]
    val_ind = rand_ind[:val_size]
    train_label = train_labelo[train_ind]
    val_label = train_labelo[val_ind]
    print(np.max(train_ind), len(trainl))

    trainl_data = trainl[train_ind]
    vall_data = trainl[val_ind]
    trains_data = trains[train_ind]
    vals_data = trains[val_ind]

    train_transforms = transforms.Compose([
        transforms.Normalize(mean=[.5], std=[.5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

    test_transforms = transforms.Compose([
        transforms.Normalize(mean=[.5], std=[.5]),
    ])

    train_dataset = MyDataset(trains_data, trainl_data, train_label, transform = train_transforms)
    valid_dataset = MyDataset(vals_data, vall_data, val_label, transform = test_transforms)
    test_dataset = MyDataset(tests_data, testl_data, test_label, transform = test_transforms)

    return train_dataset, valid_dataset, test_dataset