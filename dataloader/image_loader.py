from PIL import Image
import os.path
import torch.utils.data
import cv2
import torchvision.transforms as transforms
from get_image import get_data
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def default_image_loader(path, image_size):
    image = cv2.imread(path)
    w, h = image_size.split(',')
    if image.shape[0] > int(h) or image.shape[1] > int(w):
        inter = cv2.INTER_AREA
    else:
        inter = cv2.INTER_CUBIC
    image = cv2.resize(image, (int(w), int(h)), interpolation=inter)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, image_size, datasets, transform=None,
                 loader=default_image_loader):
        self.datasets = datasets
        self.transform = transform
        self.loader = loader
        self.image_size = image_size
    def __getitem__(self, index):
        path, label = self.datasets[index]
        img = self.loader(path, self.image_size)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.datasets)


def get_loader(args):
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

    means = (0.485, 0.456, 0.406)
    stds = (0.229, 0.224, 0.225)

    train_data, test_data, labels_class = get_data(args)

    train_data_loader = torch.utils.data.DataLoader(
        ImageLoader(args.image_size, train_data,transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(means, stds)
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_data_loader = torch.utils.data.DataLoader(
        ImageLoader(args.image_size, test_data,transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize(means, stds)
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    return train_data_loader, test_data_loader, labels_class
