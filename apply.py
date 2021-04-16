import os
import json
from skimage import io
import numpy as np


def integrate_image_signal_over_channels(masks_directory_path: str, images_path: str,
                                         target_channel_names: str, save=False, save_path: str = '.',
                                         interest_mask=None):
    """
    :interest_mask_indices: masks of interest indices splitted by commas
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
