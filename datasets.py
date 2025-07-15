import os
from pathlib import Path
from typing import Union

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as TF


def check_names(list_files):
    # check file format
    for file in list_files:
        if not file.endswith('.jpg'):
            list_files.remove(file)
    return list_files


class IHarmony4Dataset(Dataset):
    def __init__(self, base_path:Union[str, Path]):
        if type(base_path) == str:
            base_path = Path(base_path)
        self.base_path = base_path
        
        self.list_files = []
        self._get_image_paths()
        
        self.transform = TF.Compose([
            TF.Resize((256, 256)),
            TF.RandomHorizontalFlip(),
            TF.ToTensor(),
            TF.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self.transform_gray = TF.Compose([
            TF.Resize((256, 256)),
            TF.Grayscale(num_output_channels=3),
            TF.RandomHorizontalFlip(),
            TF.ToTensor(),
            TF.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return len(self.list_files)
        
    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        img_path = self.list_files[idx]
        img = Image.open(self.base_path/img_path).convert('RGB')
        img_tensor = self.transform(img)
        img_gray = self.transform_gray(img)
        return img_tensor, img_gray, img_path[:-4]

    def _get_image_paths(self):
        list_subdatasets = ['HAdobe5k', 'Hday2night', 'HFlickr', 'HCOCO']
        list_total_files = []
        for subdataset in list_subdatasets:
            list_files = os.listdir(self.base_path / subdataset/'real_images')
            list_files = check_names(list_files)
            list_files = [subdataset+'/'+'real_images'+'/'+file for file in list_files]
            list_total_files += list_files
        self.list_files = list_total_files