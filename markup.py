from train_utils import DataLoaderRandomSquare, binarize_image
import torch
import os
from skimage import io


def markup(model: torch.nn.Module, markup_images_path: str, target_channel_names: str, save_path: str, device='cpu',
           if_binary_mask=True):

    target_channels_list = target_channel_names.split(',')
    target_channels_paths = [os.path.join(markup_images_path, channel) for channel in target_channels_list]

    markup_loader = DataLoaderRandomSquare(images_paths=target_channels_paths, random_square=False, augmentate=False)
    markup_batch_gen = torch.utils.data.DataLoader(markup_loader, batch_size=1, shuffle=False)

    images_names = markup_loader.images[0]
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, image in zip(images_names, markup_batch_gen):
        pred_mask = model(image.to(device)).cpu().data.numpy().reshape(*image.shape[-2:])
        if if_binary_mask:
            pred_mask = binarize_image(pred_mask)
        image_save_path = os.path.join(save_path, name)
        io.imsave(image_save_path, pred_mask)
