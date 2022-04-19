from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset

class SIGFDataset(Dataset):
    def __init__(self, root, tsfm=None):
        super().__init__()
        self.root = root
        self.img_paths = list(self.root.rglob('*.JPG'))
        self.tsfm = tsfm

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        if self.tsfm is not None:
            img = self.tsfm(img)
        return {'image': img, 'label': -1}

class SIGFDatasetTrain(SIGFDataset):
    def __init__(self, size, degradation):
        
        root = Path('/home/niklas/projects/latent-diffusion/SIGF-database/train/image')
        tsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        super().__init__(root, tsfm)


class SIGFDatasetValid(SIGFDataset):
    def __init__(self, size, degradation):
        
        root = Path('/home/niklas/projects/latent-diffusion/SIGF-database/validation/image')
        tsfm = transforms.Compose([
            transforms.Resize(256),
            transforms.ToTensor()
        ])
        super().__init__(root, tsfm)