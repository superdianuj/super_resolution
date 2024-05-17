# Case Study: Train the model on some custom dataset with manual resolution adjustments, and then test the trained model on a new but similar image


| Ref High Resolution Dataset | SRCNN Predictions from Low Resolution (0.5 x) |
|---------|------------------|
| ![GIF 1](super_res1/highres.gif)<br>Used in Training | ![GIF 2](super_res1/highres_pred.gif)<br>Prediction |
| ![GIF 3](super_res1/new_highres.gif)<br>Not used in Training | ![GIF 4](super_res1/new_highres_pred.gif)<br>Prediction |


# Case Study: Using a pretrained model Nina_SR model (trained on Div2K dataset on 300 epochs), and testing on new images

| Ref High Resolution Dataset | Nina_SR Predictions from Low Resolution (0.5 x)|
|---------|------------------|
| ![GIF 1](super_res2/vase_highres.gif)<br>Not used in Training | ![GIF 2](super_res2/vase_highres_pred.gif)<br>Prediction |
| ![GIF 3](super_res2/balcony_highres.gif)<br>Not used in Training | ![GIF 4](super_res2/balcony_highres_pred.gif)<br>Prediction |


# Case Study: Improving Resolution Improves 3D rendering result (COLMAP Dense Reconstruction Example)

## Low Resolution Dataset


https://github.com/superdianuj/super_resolution/assets/47445756/9530f366-6f7a-40c8-98c2-55d863006865



## High Resolution Dataset (x2)


https://github.com/superdianuj/super_resolution/assets/47445756/1c2a503e-e6e6-4490-b6d9-ca0b4672a9e1




# Reference
https://github.com/isaaccorley/pytorch-enhance

https://github.com/Coloquinte/torchSR
