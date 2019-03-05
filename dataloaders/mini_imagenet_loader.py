import os.path as osp
from PIL import Image

from torch.utils.data import Dataset
from torchvision import transforms


ROOT_PATH = './mini_imagenet/'


class MiniImageNet(Dataset):

    def __init__(self, setname):
        csv_path = osp.join(ROOT_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []
        self.class_idx_to_sample_idx = {}

        for idx, l in enumerate(lines):
            name, wnid = l.split(',')
            path = osp.join(ROOT_PATH, 'images', name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
                self.class_idx_to_sample_idx.update({lb: []})
            data.append(path)
            label.append(lb)
            self.class_idx_to_sample_idx[lb].append(idx)

        self.data = data
        self.labels = label

        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.labels[i]
        image = self.transform(Image.open(path).convert('RGB'))
        return image, label
