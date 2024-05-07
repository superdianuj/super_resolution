import torch
from torchsr.datasets import RealSRv3
from torchsr.models import ninasr_b0
from torchvision.transforms.functional import to_pil_image, to_tensor
import cv2
import numpy as np
import os
import argparse

parser = argparse.ArgumentParser(description='Super Resolution')
parser.add_argument('--img_size', type=int, default=256, help='Image size')
parser.add_argument('--res_scale', type=int, default=2, help='Resolution scale')
parser.add_argument('--dir', type=str, default='new_images', help='Directory of images')
args=parser.parse_args()

IMG_SIZE=args.img_size
res_scale=args.res_scale

# # Download a pretrained NinaSR model
model = ninasr_b0(scale=res_scale, pretrained=True)

#--------------------------------------------------------
#--------------------------------------------------------

# img_name= 'ref_pic.jpg'
# name_of_image=img_name.split('.')[0]
# img=cv2.imread(img_name)
# img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img=cv2.resize(img, (IMG_SIZE,IMG_SIZE))




# img_tens=np.transpose(img,(2,0,1))
# img_tens=torch.from_numpy(img_tens)
# img_tens=img_tens.unsqueeze(0)
# img_tens=img_tens/255.0
# img_tens=img_tens.float()


# sr_t = model(img_tens)
# sr = to_pil_image(sr_t.squeeze(0).clamp(0, 1))
# sr.show()

# # sr.to_pil_image().save('high_res'+name_of_image+'.png')
# sr2=to_pil_image(img_tens.squeeze(0).clamp(0, 1))
# # save the high res image
# sr.save('high_res'+name_of_image+'.png')

#--------------------------------------------------------
#--------------------------------------------------------




# load images in images directory
dir=args.dir
file_names = sorted(os.listdir(dir), key=lambda x: int(x.split('_')[-1].split('.')[0]) if '_' in x else int(x.split('.')[0]))
images_path = [os.path.join(dir, file_name) for file_name in file_names if file_name.endswith('.JPG') or file_name.endswith('.png') or file_name.endswith('.jpg')]
images = [cv2.imread(image_path) for image_path in images_path]
dum=np.array(images).shape


images_ref = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

# images_ref = [np.transpose(img,(2,0,1)) for img in images_ref]
images_l = [cv2.resize(image, (IMG_SIZE, IMG_SIZE),interpolation=cv2.INTER_AREA) for image in images_ref]
images_h = [cv2.resize(image, (IMG_SIZE*res_scale, IMG_SIZE*res_scale),interpolation=cv2.INTER_AREA) for image in images_ref]

images_l = torch.tensor(np.array(images_l))
images_h = torch.tensor(np.array(images_h))

images_l = images_l.permute(0, 3, 1, 2)/255.0
images_h = images_h.permute(0, 3, 1, 2)/255.0

images_l=images_l.float()
images_h=images_h.float()



if os.path.exists(dir+'_res'):
    os.system('rm -rf '+dir+'_res')

if os.path.exists(dir+'_highres'):
    os.system('rm -rf '+dir+'_highres')

if os.path.exists(dir+'_highres_pred'):
    os.system('rm -rf '+dir+'_highres_pred')    

if not os.path.exists(dir+'_res'):
    os.makedirs(dir+'_res')

if not os.path.exists(dir+'_highres'):
    os.makedirs(dir+'_highres')

if not os.path.exists(dir+'_highres_pred'):
    os.makedirs(dir+'_highres_pred')

for i in range(len(images_l)):

    curr_img_l=images_l[i:i+1]
    # flip channels
    ref_img=images_h[i:i+1]
    curr_img_h=model(curr_img_l)

    curr_img_l=torch.permute(curr_img_l.detach().cpu(), (0, 2, 3, 1)).numpy()[0]
    curr_img_l=((curr_img_l)*255.0).astype(np.uint8)
    curr_img_l=curr_img_l[...,::-1]
    cv2.imwrite(dir+'_res'+'/'+file_names[i], curr_img_l)

    curr_img_h=torch.permute(curr_img_h.detach().cpu(), (0, 2, 3, 1)).numpy()[0]
    curr_img_h=((curr_img_h)*255.0).astype(np.uint8)
    curr_img_h=curr_img_h[...,::-1]
    cv2.imwrite(dir+'_highres'+'/'+file_names[i], curr_img_h)

    ref_img=torch.permute(ref_img.detach().cpu(), (0, 2, 3, 1)).numpy()[0]
    ref_img=((ref_img)*255.0).astype(np.uint8)
    ref_img=ref_img[...,::-1]
    cv2.imwrite(dir+'_highres_pred'+'/'+file_names[i], ref_img)
