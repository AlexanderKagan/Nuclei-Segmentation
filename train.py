from Unet import UNet
from train_utils import train_loop, DataLoaderRandomSquare
from image_utils import masks_from_coco_json, make_val_directory
from albumentations import Compose, RandomRotate90, Transpose, VerticalFlip

import torch
import json
import os
import argparse
import imageio.core.util


AUGMENTATION = Compose([
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Transpose(p=0.5)]
              )


def silence_imageio_warning(*args, **kwargs):
    pass


imageio.core.util._precision_warn = silence_imageio_warning


def train(coco_json_path: str, train_images_path: str, target_channel_names: str, train_directory: str = '.',
          validate=False, device='cpu', num_epochs=50, model_name='unet', init_features=10,
          train_image_width=128, train_image_height=128, **kwargs):
    """
    :train_images_path: path to the directory consisting of the folders with channel corresponding images, e.g.
    train_images_path/lipid/...(lipid images)', 'train_images_path/protein/...(protein images)'
    :target_channel_names: string with target channel names splitted by commas WITHOUT SPACES
    Names should be the same as the folders in train_images_path, e.g. ('lipid,protein,dapi')
    :train_directory: path to the directory where all training data will be saved (val_masks, val_images, model, e.t.c)
    :device: 'cpu' or 'cuda' - device on which the model will be trained
    :init_features: number of filters used in UNet architecture (for more info look at the architecture in Unet.py)

    1) Reads COCO json from coco_json_path and saves parsed masks to 'train_directory/train_masks'
    2) If validate==True makes 'train_directory/val_images' and 'train_directory/val_masks' directories with fixed crops
    of size (train_image_width, train_image_height) for each train image.
    3) Trains the model on train_images and train_masks possibly with validation part on val_images and val_masks.
    The model is trained on random crops of size (train_image_width, train_image_height) of images in train_images_path
    4) Saves the model with name model_name to train_directory
    """
    with open(coco_json_path, 'r') as f:
        coco_json = json.load(f)

    train_masks_save_path = os.path.join(train_directory, 'train_masks/')
    _ = masks_from_coco_json(coco_json, save=True, save_path=train_masks_save_path)

    target_channels_list = target_channel_names.split(',')
    target_channels_paths = [os.path.join(train_images_path, channel) for channel in target_channels_list]

    if not os.path.exists(train_directory):
        os.makedirs(train_directory)

    if validate:
        val_images_directory_path = os.path.join(train_directory, 'val_images/')
        val_masks_directory_path = os.path.join(train_directory, 'val_masks/')

        make_val_directory(images_save_dir=val_images_directory_path, images_to_cut_path=train_images_path,
                           masks_save_dir=val_masks_directory_path, masks_to_cut_path=train_masks_save_path,
                           target_channel_names=target_channels_list,
                           width=train_image_width, height=train_image_height)

        val_images_paths_list = [os.path.join(val_images_directory_path, channel) for channel in target_channels_list]
        val_loader = DataLoaderRandomSquare(images_paths=val_images_paths_list,
                                            masks_path=val_masks_directory_path,
                                            random_square=False,
                                            width=train_image_width, height=train_image_height
                                            )
    else:
        val_loader = None

    train_loader = DataLoaderRandomSquare(images_paths=target_channels_paths,
                                          masks_path=train_masks_save_path,
                                          random_square=True,
                                          width=train_image_width, height=train_image_height,
                                          augmentate=True, augmentation=AUGMENTATION)

    model = UNet(in_channels=len(target_channels_list), out_channels=1, init_features=init_features)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda epoch: 0.99 ** epoch)

    _ = train_loop(model=model, opt=opt, sheduler=sheduler,
                   train_loader=train_loader, val_loader=val_loader, validate=validate,
                   device=device, num_epochs=num_epochs,
                   save=True, save_name=os.path.join(train_directory, model_name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--coco_json_path', action='store', type=str, required=True)
    parser.add_argument('--train_images_path', action='store', type=str, required=True)
    parser.add_argument('--target_channel_names', action='store', type=str, required=True)
    parser.add_argument('--validate', action='store', type=bool, default=False)
    parser.add_argument('--device', action='store', type=str, default='cpu')
    parser.add_argument('--num_epochs', action='store', type=int, default=50)
    parser.add_argument('--init_features', action='store', type=int, default=10)

    args = parser.parse_args()

    train(**vars(args))
