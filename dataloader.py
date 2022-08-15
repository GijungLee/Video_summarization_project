import os
import glob
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from natsort import natsorted

class ImageDataset(Dataset):
    def __init__(self, root, transform1=None, class_name="fish-big"):
        self.transform1 = transform1
        self.files1 = natsorted(sorted(glob.glob(os.path.join(root, "Images", class_name) + "/*.jpg")))

    def __getitem__(self, index):
        img_A = Image.open(self.files1[index % len(self.files1)])
        img_A = self.transform1(img_A)


        return {"A": img_A}

    def __len__(self):
        return len(self.files1)

train_transform1 = transforms.Compose([transforms.ToTensor(),
                                      transforms.Resize((256, 256), transforms.InterpolationMode.BICUBIC),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
