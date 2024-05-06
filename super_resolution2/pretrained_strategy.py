import torch
from torchsr.datasets import RealSRv3
from torchsr.models import ninasr_b0
from torchvision.transforms.functional import to_pil_image, to_tensor
import cv2
import numpy as np


IMG_SIZE=256

# Div2K dataset
dataset = RealSRv3(root="./datasets", scale=2, download=False)

# Get the first image in the dataset (High-Res and Low-Res)
hr, lr = dataset[0]

# Download a pretrained NinaSR model
model = ninasr_b0(scale=2, pretrained=True)

img_name= 'ref_pic.jpg'
name_of_image=img_name.split('.')[0]
img=cv2.imread(img_name)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img=cv2.resize(img, (IMG_SIZE,IMG_SIZE))

img_tens=np.transpose(img,(2,0,1))
img_tens=torch.from_numpy(img_tens)
img_tens=img_tens.unsqueeze(0)
img_tens=img_tens/255.0
img_tens=img_tens.float()


# Run the Super-Resolution model
# lr_t = to_tensor(img_tens).unsqueeze(0)
# print(lr_t.shape)
# sldf


sr_t = model(img_tens)
sr = to_pil_image(sr_t.squeeze(0).clamp(0, 1))
sr.show()

# sr.to_pil_image().save('high_res'+name_of_image+'.png')
sr2=to_pil_image(img_tens.squeeze(0).clamp(0, 1))
sr2.show()

# sr2.to_pil_image().save('low_res'+name_of_image+'.png')