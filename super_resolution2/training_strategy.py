import torch
import torch.nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import pytorch_lightning as pl

from torch_enhance.datasets import BSDS300, Set14, Set5

from torch_enhance.models import SRCNN
from torch_enhance import metrics

import torch
import torch_enhance
import matplotlib.pyplot as plt
import cv2

import numpy as np
import os

import argparse
from torchvision.transforms.functional import to_pil_image, to_tensor


class Dataseter(Dataset):
    def __init__(self, images, images_h):
        self.images = images
        self.images_h = images_h

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.images_h[idx]
    



class Module(pl.LightningModule):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")
        
        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        self.log("train_loss", loss)
        self.log("train_mae", mae)
        self.log("train_psnr", psnr)

        return loss

    def validation_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")
        
        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        self.log("val_loss", loss)
        self.log("val_mae", mae)
        self.log("val_psnr", psnr)

        return loss

    def test_step(self, batch, batch_idx):
        lr, hr = batch
        sr = self(lr)
        loss = F.mse_loss(sr, hr, reduction="mean")
        
        # metrics
        mae = metrics.mae(sr, hr)
        psnr = metrics.psnr(sr, hr)

        # Logs
        self.log("test_loss", loss)
        self.log("test_mae", mae)
        self.log("test_psnr", psnr)

        return loss


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='SRCNN')
    parser.add_argument('--img_size', type=int, default=256, help='original image size')
    parser.add_argument('--scale_factor', type=int, default=2, help='scale factor for resolution')
    parser.add_argument('--bs', type=int, default=32, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    args=parser.parse_args()

 
    IMG_SIZE=args.img_size
    scale_factor=args.scale_factor
    IMG_SIZE_H=IMG_SIZE*scale_factor
    BS=args.bs
    NUM_EPOCHS=args.num_epochs


    # load images in images directory
    dir='images'
    file_names = sorted(os.listdir(dir), key=lambda x: int(x.split('.')[0]) if '_' in x else int(x.split('.')[0]))
    images_path = [os.path.join(dir, file_name) for file_name in file_names if file_name.endswith('.jpg') or file_name.endswith('.png')]
    images = [cv2.imread(image_path) for image_path in images_path]
    dum=np.array(images).shape


    images_ref = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]
    # images_ref = [np.transpose(img,(2,0,1)) for img in images_ref]
    images_l = [cv2.resize(image, (IMG_SIZE, IMG_SIZE),interpolation=cv2.INTER_AREA) for image in images_ref]
    images_h = [cv2.resize(image, (IMG_SIZE_H, IMG_SIZE_H),interpolation=cv2.INTER_AREA) for image in images_ref]


    images_l = torch.tensor(np.array(images_l))
    images_h = torch.tensor(np.array(images_h))

    images_l = images_l.permute(0, 3, 1, 2)/255.0
    images_h = images_h.permute(0, 3, 1, 2)/255.0

    images_l=images_l.float()
    images_h=images_h.float()


    dataset = Dataseter(images_l, images_h)
    dataloader = DataLoader(dataset, batch_size=BS, shuffle=True)

    # Define model
    channels = 3
    model = SRCNN(scale_factor, channels)
    module = Module(model)

    trainer = pl.Trainer(max_epochs=NUM_EPOCHS, accelerator="gpu")

    trainer.fit(
        module,
        dataloader,
    )

    trained_model = SRCNN(scale_factor, channels)
    trained_model.load_state_dict(module.model.state_dict())

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

    i_img_tens = model(img_tens)

    plt.imshow(((torch.permute(img_tens.detach().cpu(),(0,2,3,1)).numpy()[0])*255.0).astype(np.uint8))
    plt.title("Low Resolution")
    plt.savefig('low_res'+name_of_image+'.png')

    plt.imshow(((torch.permute(i_img_tens.detach().cpu(),(0,2,3,1)).numpy()[0])*255.0).astype(np.uint8))
    plt.title("High Resolution")
    plt.savefig('high_res'+name_of_image+'.png')
    # plt.show()

    

    sr = to_pil_image(img_tens.squeeze(0).clamp(0, 1))
    sr.show()

    # sr.to_pil_image().save('high_res'+name_of_image+'.png')
    sr2=to_pil_image(i_img_tens.squeeze(0).clamp(0, 1))
    sr2.show()


    if os.path.exists('lowres'):
        os.system('rm -rf lowres')

    if os.path.exists('highres'):
        os.system('rm -rf highres')
    
    if os.path.exists('highres_pred'):
        os.system('rm -rf highres_pred')    

    if not os.path.exists('lowres'):
        os.makedirs('lowres')

    if not os.path.exists('highres'):
        os.makedirs('highres')

    if not os.path.exists('highres_pred'):
        os.makedirs('highres_pred')

    for i in range(len(images_l)):

        curr_img_l=images_l[i:i+1]
        ref_img=images_h[i:i+1]
        curr_img_h=model(curr_img_l)

        curr_img_l=torch.permute(curr_img_l.detach().cpu(), (0, 2, 3, 1)).numpy()[0]
        curr_img_l=((curr_img_l)*255.0).astype(np.uint8)
        cv2.imwrite('lowres/'+file_names[i], curr_img_l)

        curr_img_h=torch.permute(curr_img_h.detach().cpu(), (0, 2, 3, 1)).numpy()[0]
        curr_img_h=((curr_img_h)*255.0).astype(np.uint8)
        cv2.imwrite('highres_pred/'+file_names[i], curr_img_h)

        ref_img=torch.permute(ref_img.detach().cpu(), (0, 2, 3, 1)).numpy()[0]
        ref_img=((ref_img)*255.0).astype(np.uint8)
        cv2.imwrite('highres/'+file_names[i], ref_img)


    print("Done!")






