import torch
import os
import skimage
from skimage import io
import numpy as np
from torchvision import transforms
import time
from copy import copy
from typing import List
from torch.utils import data

# to avoid hidden files such as DS_Store
IS_IMAGE = lambda img: img[img.rfind('.') + 1:] in ('jpg', 'png', 'tif', 'bmp')


class DataLoaderRandomSquare(data.Dataset):
    def __init__(self,
                 images_paths: List[str],
                 masks_path: str = None,
                 width=128, height=128,
                 augmentate=False, augmentation=None,
                 random_square=True):

        self.random_square = random_square
        self.width = width
        self.height = height

        self.images = [sorted([img for img in os.listdir(images_path) if IS_IMAGE(img)])
                       for images_path in images_paths]
        if masks_path is not None:
            self.masks = sorted([img for img in os.listdir(masks_path) if IS_IMAGE(img)])

        self.images_paths = images_paths
        self.masks_path = masks_path

        self.augmentate = augmentate
        self.augmentation = augmentation

    def __len__(self):
        return len(self.images[0])

    def __getitem__(self, image_index: int, totensor=True):
        image_layers = []
        for i, images_path in enumerate(self.images_paths):
            image_layer_path = os.path.join(images_path, self.images[i][image_index])
            image_layer = np.asarray(io.imread(image_layer_path), dtype=np.float32)
            image_layers.append(image_layer if len(image_layer.shape) == 3 else image_layer[:, :, None])
        image = np.concatenate(image_layers, -1)
        if self.masks_path is not None:
            mask = skimage.img_as_ubyte(io.imread(os.path.join(self.masks_path, self.masks[image_index])))

        if self.random_square:
            y_left_upper_corner = np.random.randint(0, image.shape[0] - self.height)
            x_left_upper_corner = np.random.randint(0, image.shape[1] - self.width)
            image = image[y_left_upper_corner: y_left_upper_corner + self.height,
                    x_left_upper_corner: x_left_upper_corner + self.width]
            if self.masks_path is not None:
                mask = mask[y_left_upper_corner: y_left_upper_corner + self.height,
                   x_left_upper_corner: x_left_upper_corner + self.width]

        if self.augmentate and self.augmentation is not None:
            aug = self.augmentation(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        if not totensor:
            if self.masks_path is not None:
                return image, mask
            return image

        transformer = transforms.Compose([transforms.ToTensor()])
        if self.masks_path is not None:
            return transformer(image), transformer(mask)
        return transformer(image)


def binarize_image(img, fill_with=1., treshold=0.5):
    image = copy(img)
    image[image > treshold] = fill_with
    image[image <= treshold] = 0
    return image


def predict_val(val_batch_gen, model, device):
    mask_2_pred = {}
    for (images, masks) in val_batch_gen:
        pred_mask = model(images.to(device)).cpu().data.numpy().reshape(images.shape[0], *images.shape[-2:])
        bin_pred_mask = binarize_image(pred_mask)
        mask_2_pred[masks] = bin_pred_mask
    return mask_2_pred


def dice_loss(y_true, y_pred, beta=1.):
    numerator = (1 + beta) * (y_true * y_pred).sum()
    denominator = (y_true + beta * y_pred).sum()
    return 1 - (numerator + 1) / (denominator + 1)


def calc_iou(prediction, ground_truth):
    n_images = len(prediction)
    intersection, union = 0, 0
    for i in range(n_images):
        intersection += np.logical_and(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
        union += np.logical_or(prediction[i] > 0, ground_truth[i] > 0).astype(np.float32).sum()
    return float(intersection) / union


def train_loop(model,
               opt: torch.optim,
               train_loader,
               val_loader=None,
               sheduler=None,
               batch_size=2,
               dice_loss_beta=1.,
               num_epochs=60,
               save=False,
               validate=True,
               device='cpu',
               save_name='unet'):

    training_scores = []
    train_batch_gen = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle=True)

    if validate and val_loader is not None:
        validation_scores = []
        val_batch_gen = torch.utils.data.DataLoader(val_loader, batch_size=batch_size, shuffle=False)

    for epoch in range(num_epochs):
        print(f'Epoch number: {epoch}')
        start_time = time.time()
        model.train()
        epoch_train_loss = []
        for (X_image, X_mask) in train_batch_gen:
            X_image = X_image.to(device)
            pred_mask = model(X_image)[:, 0].contiguous().view(-1)

            true_mask = X_mask[:, 0].contiguous().view(-1).to(device)
            loss = dice_loss(true_mask, pred_mask, dice_loss_beta)
            loss.backward()
            opt.step()
            opt.zero_grad()
            if sheduler is not None:
                sheduler.step()
            epoch_train_loss.append(loss.data.cpu().numpy())

        training_scores.append(np.mean(epoch_train_loss))
        if validate:
            model.eval()
            masks_2_pred = predict_val(val_batch_gen, model, device)
            val_preds = np.vstack(list(masks_2_pred.values()))
            val_true = np.vstack(list(masks_2_pred.keys()))
            val_iou = calc_iou(val_preds, val_true)
            validation_scores.append(val_iou)
            print('validation iou is {}'.format(val_iou))
            print(f'Training epoch loss: {training_scores[-1]}')
            print("Epoch {} of {} took {:.3f}s".format(epoch + 1, num_epochs, time.time() - start_time))
    if save:
        torch.save(model.state_dict(), f'./{save_name}')

    return (training_scores, validation_scores) if validate else training_scores
