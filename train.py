from Unet import UNet
from train_utils import train_loop, DataLoaderRandomSquare
from image_utils import masks_from_coco_json, make_val_directory
from albumentations import Compose, RandomRotate90, Transpose, VerticalFlip

import torch
import json
import os

AUGMENTATION = Compose([
                VerticalFlip(p=0.5),
                RandomRotate90(p=0.5),
                Transpose(p=0.5)]
              )


def train(coco_json_path: str, train_images_path: str, target_channel_names: str, train_directory: str,
          validate=False, device='cpu', num_epochs=60, model_name='unet', init_features=20,
          train_image_width=128, train_image_height=128):
    """
    :target_channel_names: string with target chanel names splitted by commas WITHOUT SPACES
    Reads COCO json from coco_json_path, saves parsed masks to 'images_path/masks' and
    then trains the model on images and masks.
    """
    with open(coco_json_path, 'r') as f:
        coco_json = json.load(f)

    train_masks_save_path = os.path.join(train_directory, 'train_masks/')
    _ = masks_from_coco_json(coco_json, save=True, save_path=train_masks_save_path)

    target_channels_list = target_channel_names.split(',')
    target_channels_paths = [os.path.join(train_images_path, channel) for channel in target_channels_list]

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
                   device=device, num_epochs=num_epochs, save=True, save_name=model_name)

    return model
