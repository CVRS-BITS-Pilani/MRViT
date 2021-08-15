import torch
import numpy as np
import torch.utils.data as data
import h5py
import torchvision.transforms as transforms

train_small_transforms = transforms.Compose([
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
    ])

train_large_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

test_small_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

test_large_transforms = transforms.Compose([
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

class MyDataset(data.Dataset):
    """dataset class for returning both small and large images along with labels."""
    def __init__(self, images_small, images_large, labels, small_transform=None, 
                    large_transform=None):
        self.images_small = images_small
        self.images_large = images_large
        self.labels = labels
        self.small_transform = small_transform
        self.large_transform = large_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_s = self.images_small[index].permute(2, 1, 0)
        image_l = self.images_large[index].permute(2, 1, 0)

        if image_s.size(0) == 1:
            image_s = image_s.repeat(3, 1, 1)
            image_l = image_l.repeat(3, 1, 1)
        if self.small_transform is not None:
            image_s = self.small_transform(image_s)
        if self.large_transform is not None:
            image_l = self.large_transform(image_l)

        label = torch.as_tensor(self.labels[index]).to(torch.int64)

        return image_s, image_l , label

    def get_labels(self): 
        return self.labels.flatten()

def get_datasets(data_flag):
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
    val_size = int(0.001 * train_size)

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

    train_dataset = MyDataset(trains_data, trainl_data, train_label, train_small_transforms, 
                                train_small_transforms)
    valid_dataset = MyDataset(vals_data, vall_data, val_label, test_small_transforms, 
                                test_small_transforms)
    test_dataset = MyDataset(tests_data, testl_data, test_label, test_small_transforms, 
                                test_small_transforms)

    return train_dataset, valid_dataset, test_dataset