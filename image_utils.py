from skimage import io
from PIL import Image, ImageDraw
import numpy as np
from math import ceil
from itertools import product
import os
import warnings


def masks_from_coco_json(coco_json, save=True, save_path='./all_masks/'):
    image_id_2_file_name = {im['id']: im['file_name'] for im in coco_json['images']}
    image_id_2_size = {im['id']: (im['height'], im['width']) for im in coco_json['images']}
    id_2_image = {img_id: Image.new('1', size=image_id_2_size[img_id]) for img_id in image_id_2_file_name.keys()}
    for annotation in coco_json['annotations']:
        img_id = annotation['image_id']
        drawer = ImageDraw.Draw(id_2_image[img_id])
        polygoon_annots = annotation['segmentation'][0]
        # coco segmentation polygon coordinates are in list format [[x1, y1, x2, y2, ...]]
        drawer.polygon(list(zip(polygoon_annots[::2], polygoon_annots[1::2])), fill=1)
    filename_2_array_image = {image_id_2_file_name[img_id]: np.array(image, dtype=int)
                              for img_id, image in id_2_image.items()}
    if save:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        for filename, array_image in filename_2_array_image.items():
            io.imsave(save_path + filename, array_image)
    return filename_2_array_image


def cut_image_into_rectangulars(image_path, width=128, height=128,
                                frmt='tif', save=True,
                                save_path='./cut_images/',
                                rectangular_indices=None):
    image = np.asarray(Image.open(image_path), dtype=np.int16)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    rectangulars = []
    H, W = image.shape if len(image.shape) == 2 else image.shape[:-1]
    for i, j in product(range(ceil(H / height)), range(ceil(W / width))):
        cut_image = image[min(H - height, height * i): min(height * (i + 1), H),
                    min(W - width, width * j): min(width * (j + 1), W)]
        rectangulars.append(cut_image)
    if save:
        save_path += image_path[image_path.rfind('/') + 1: image_path.rfind('.')]
        if rectangular_indices is None:
            rectangular_indices = range(len(rectangulars))

        for rect_idx in rectangular_indices:
            io.imsave(save_path + '_{}.{}'.format(rect_idx, frmt), rectangulars[rect_idx])

    return rectangulars


def make_val_directory(images_save_dir, images_to_cut_path: str, target_channel_names: list,
                       masks_save_dir: str, masks_to_cut_path: str,
                       width=128, height=128, save_format='tif'):
    """
    :images_to_cut_path: path to folder where chanel corresponding folders with images are kept.
    :target_chanel_names: list of folers names corresponding to chanels on which model will be validated
    CHANEL NAMES SHOULD BE THE SAME AS CORRESPONDING SUBDIRECTORIES IN images_to_cut_path
    """

    for folder_name in target_channel_names:
        folder_path = os.path.join(images_to_cut_path, folder_name)
        for img_name in os.listdir(folder_path):
            try:
                cut_image_into_rectangulars(f'{images_to_cut_path}/{folder_name}/{img_name}',
                                            save_path=f'{images_save_dir}/{folder_name}/',
                                            width=width, height=height, frmt=save_format, save=True)
            except:
                warnings.warn('Not only image fsiles in target images folder', ImportWarning)

    for mask_name in os.listdir(masks_to_cut_path):

        try:
            cut_image_into_rectangulars(f'{masks_to_cut_path}/{mask_name}',
                                        save_path=f'{masks_save_dir}/',
                                        width=width, height=height, frmt=save_format)
        except:
            warnings.warn('Not only image files in target images folder', ImportWarning)
