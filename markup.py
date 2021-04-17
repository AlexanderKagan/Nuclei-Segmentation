from train_utils import DataLoaderRandomSquare, binarize_image
from Unet import UNet

import torch
import os
from skimage import io
import argparse


def markup(model_path: str, markup_images_path: str, target_channel_names: str, save_path: str = './markups', device='cpu',
           if_binary_mask=True):
    """
    Predicts and saves the masks for given images using the trained UNet
    :model_path: path to the state dict file of the trained model (by now works only for UNet architecture!)
    :markup_images_path: path to the images for which we want the masks to be produced. markup_images_path should
    consist of channel corresponding folders, e.g. markup_images_path/lipid/...(lipid images).
    :target_channel_names: string consisting of channel names splittted by commas WITHOUT SPACES.
    To work properly all of channels should be:
    1) among the channel folders names in markup_images_path
    2) the same as the ones on which the model was trained
    :save_path: path to the directory to which the marked up images (masks) should be saved.
    :device: 'cpu' or 'cuda' - device on which to make the markup
    :if_binary_mask: if True makes the binary markup (all pixels of the mas are either 1, or 0), if False
    saves the masks with probabilities 0<=p_ij<=1 as in the output of the model.
    """

    target_channels_list = target_channel_names.split(',')
    target_channels_paths = [os.path.join(markup_images_path, channel) for channel in target_channels_list]

    markup_loader = DataLoaderRandomSquare(images_paths=target_channels_paths, random_square=False, augmentate=False)
    markup_batch_gen = torch.utils.data.DataLoader(markup_loader, batch_size=1, shuffle=False)

    # Did not find a better way to recover the number of init_features from the state_dict
    state_dict = torch.load(model_path)
    init_features = state_dict['encoder1.enc1conv1.weight'].shape[0]
    model = UNet(in_channels=len(target_channels_list), out_channels=1, init_features=init_features)
    model.load_state_dict(state_dict)

    images_names = markup_loader.images[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, image in zip(images_names, markup_batch_gen):
        pred_mask = model(image.to(device)).cpu().data.numpy().reshape(*image.shape[-2:])
        if if_binary_mask:
            pred_mask = binarize_image(pred_mask)
        image_save_path = os.path.join(save_path, name)
        io.imsave(image_save_path, pred_mask)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model_path', action='store', type=str, required=True)
    parser.add_argument('--markup_images_path', action='store', type=str, required=True)
    parser.add_argument('--target_channel_names', action='store', type=str, required=True)
    parser.add_argument('--if_binary_mask', action='store', type=bool, default=True)
    parser.add_argument('--device', action='store', type=str, default='cpu')
    parser.add_argument('--save_path', action='store', type=str, default='./markups')

    args = parser.parse_args()
    markup(**vars(args))
