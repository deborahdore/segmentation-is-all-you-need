import cv2
import numpy as np
from torch.utils.data import random_split, DataLoader
from torchvision import datasets
from torchvision.transforms import transforms


class BaseDataLoader:
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=8):
        self.data_dir = data_dir
        self.batch_size = batch_size

        transformer = transforms.Compose([transforms.CenterCrop((900, 1600)),
                                          transforms.Resize((135, 240)),
                                          transforms.ToTensor()])

        folder = datasets.ImageFolder(data_dir, transform=transformer)

        train_set, test_set = random_split(folder, [0.8, 0.2])

        self.train_loader = DataLoader(train_set,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=True)
        self.test_loader = DataLoader(test_set,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      pin_memory=True)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader


class ValidDataLoader:
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=8):
        self.data_dir = data_dir
        self.batch_size = batch_size

        transformer = transforms.Compose([transforms.CenterCrop((900, 1600)),
                                          transforms.Resize((135, 240)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor()])

        folder = datasets.ImageFolder(data_dir, transform=transformer)

        train_set, test_set = random_split(folder, [0.8, 0.2])
        train_set, valid_set = random_split(train_set, [0.8, 0.2])

        self.train_loader = DataLoader(train_set,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=True)
        self.test_loader = DataLoader(test_set,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      pin_memory=True)
        self.val_loader = DataLoader(valid_set,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=True)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_val_loader(self):
        return self.val_loader


class KMeansDataLoader:
    def __init__(self, data_dir, batch_size, shuffle=True, num_workers=8):
        self.data_dir = data_dir
        self.batch_size = batch_size

        transformer = transforms.Compose([transforms.CenterCrop((900, 1600)),
                                          transforms.Resize((135, 240)),
                                          transforms.RandomHorizontalFlip(),
                                          KMeansTransformer(),
                                          transforms.ToTensor()])

        folder = datasets.ImageFolder(data_dir, transform=transformer)

        train_set, test_set = random_split(folder, [0.8, 0.2])
        train_set, valid_set = random_split(train_set, [0.8, 0.2])

        self.train_loader = DataLoader(train_set,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       num_workers=num_workers,
                                       pin_memory=True)
        self.test_loader = DataLoader(test_set,
                                      batch_size=batch_size,
                                      shuffle=shuffle,
                                      num_workers=num_workers,
                                      pin_memory=True)
        self.val_loader = DataLoader(valid_set,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers,
                                     pin_memory=True)

    def get_train_loader(self):
        return self.train_loader

    def get_train_dataset_size(self):
        return len(self.train_loader.dataset)

    def get_test_loader(self):
        return self.test_loader

    def get_test_dataset_size(self):
        return len(self.test_loader.dataset)

    def get_val_loader(self):
        return self.val_loader

    def get_val_dataset_size(self):
        return len(self.val_loader.dataset)


class KMeansTransformer:
    def __call__(self, image):
        image = np.array(image)

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        vectorized = np.float32(img.reshape((-1, image.shape[2])))

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(vectorized, 5, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]

        res = res.reshape(image.shape[0], image.shape[1], image.shape[2]).astype("float32")

        return res

    def __repr__(self):
        return self.__class__.__name__ + '()'
