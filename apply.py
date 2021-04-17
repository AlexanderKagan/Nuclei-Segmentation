import os
import json
from skimage import io
import numpy as np
import argparse


def integrate_image_signal_over_channels(masks_directory_path: str, images_path: str,
                                         target_channel_names: str, save=False, save_path: str = '.',
                                         interest_mask=None):
    """
    For each pair of mask in masks_directory_path and image channel in images_path/target_channel_name integrates the
    channel over the mask, i.e. calculates the sum of pixels in the channel over all corresponding non-zero pixels in
    the mask.

    :masks_directory_path: path to masks directory
    :images_path: path to channel corresponding folders with images
    :target_channel_names: string of channel names spliited by commas WITHOUT SPACES for which to calculate the
    integrated signal. Should be chosen from exiting folders names in images_path/.
    :save: if True saves the json file with integrated signals to the save_path directory, if False only gives the dict
    object as an output
    :interest_mask: if not None calculates the integrated signal only for given mask name, otherwise for all masks in
    folder
    """

    masks = []
    mask_names = sorted(os.listdir(masks_directory_path))
    for mask_name in mask_names:
        mask_path = os.path.join(masks_directory_path, mask_name)
        mask = np.asarray(io.imread(mask_path), dtype=np.int16)
        masks.append(mask)

    target_channels_list = target_channel_names.split(',')
    integrated_signal_dict = {channel: {} for channel in target_channels_list}

    for channel in target_channels_list:
        channel_path = os.path.join(images_path, channel)
        images_names = sorted(os.listdir(channel_path))

        for i, (mask, image_name) in enumerate(zip(masks, images_names)):
            if interest_mask is None or (interest_mask is not None and mask_names[i] == interest_mask):
                image_path = os.path.join(channel_path, image_name)
                image = np.asarray(io.imread(image_path), dtype=np.int16)
                image_integrated_signal = (image * mask).sum()
                integrated_signal_dict[channel][image_name] = int(image_integrated_signal)
    if save:
        with open(os.path.join(save_path, 'channels_integrated_signal.json'), 'w') as file:
            json.dump(integrated_signal_dict, file)

    return integrated_signal_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='integrate each image in each target channel over the binary mask')
    parser.add_argument('--masks_directory_path', action='store', type=str, required=True)
    parser.add_argument('--images_path', action='store', type=str, required=True)
    parser.add_argument('--target_channel_names', action='store', type=str, required=True)
    parser.add_argument('--save', action='store', type=bool, default=False)
    parser.add_argument('--save_path', action='store', type=int, default='.')
    parser.add_argument('--interest_mask', action='store', type=str, default=None)

    args = parser.parse_args()
    integrate_image_signal_over_channels(**vars(args))
