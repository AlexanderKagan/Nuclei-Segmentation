# Nuclei-Segmentation
This repository provides the whole pipeline needed to train your Neural Network on the nuclei segmentation task, being only provided by train images of cells in different channels + COCO json Annotations to these images.

## General Discription
Main functions to work with are: 
1. train.py 

    1) Reads COCO json from coco_json_path and saves parsed masks to 'train_directory/train_masks'
    2) If validate==True makes 'train_directory/val_images' and 'train_directory/val_masks' directories with fixed crops
    of size (train_image_width, train_image_height) for each train image.
    3) Trains the model on train_images and train_masks possibly with validation part on val_images and val_masks.
    The model is trained on random crops of size (train_image_width, train_image_height) of images in train_images_path
    4) Saves the model with name model_name to train_directory
    
    #### Arguments:
    
     :train_images_path: str (required) Path to the directory consisting of the folders with channel corresponding images, e.g.
    train_images_path/lipid/...(lipid images)', 'train_images_path/protein/...(protein images)' 
    
    :target_channel_names: str (required) String with target channel names splitted by commas WITHOUT SPACES
    Names should be the same as the folders in train_images_path, e.g. ('lipid,protein,dapi')
    
    :train_directory: str (required) Path to the directory where all training data will be saved (val_masks, val_images, model, e.t.c)
    
    :device: str ('cpu' or 'cuda') Device on which the model will be trained
    
    :init_features: int Number of filters used in UNet architecture (for more info look at the architecture in Unet.py)
    
2. markup.py 

    Predicts and saves the masks for given images using the trained UNet model
    
    #### Arguments:
    
    :model_path: str (required) Path to the state dict file of the trained model (by now works only for UNet architecture!)
    
    :markup_images_path: str (required) Path to the images for which we want the masks to be produced. markup_images_path should
    consist of channel corresponding folders, e.g. markup_images_path/lipid/...(lipid images).
    
    :target_channel_names: str (required) String consisting of channel names splittted by commas WITHOUT SPACES.
    To work properly all of channels should be:
    1) among the channel folders names in markup_images_path
    2) the same as the ones on which the model was trained
    
3. apply.py 

    For each pair of mask in masks_directory_path and image channel in images_path/target_channel_name integrates the
    channel over the mask, i.e. calculates the sum of pixels in the channel over all corresponding non-zero pixels in
    the mask.
    
    #### Arguments:
    
    :masks_directory_path: str (required) Path to masks directory
    
    :images_path: str (required) path to channel corresponding folders with images
    
    :target_channel_names: str (required)  String of channel names spliited by commas WITHOUT SPACES for which to calculate the
    integrated signal. Should be chosen from exiting folders names in images_path/.
    
    :save: bool If True saves the json file with integrated signals to the save_path directory, if False only gives the dict
    object as an output
    
    :interest_mask: str If not None calculates the integrated signal only for given mask name, otherwise for all masks in
    masks_directory_path
    
## Terminal launching
0) Go to folder where the envirinment folder with needed python packages will be saved
1) Make new python environment (in the example called 'env'):
    ```console
    python3 -m venv env
    ```
2) Activate environment
   ```console
    source env/bin/activate
    ```
3) Go to folder with cloned repository and install all the required packages
   ```console
    pip install -r requirements.txt
    ```
4) Not leaving the cloned repo directory train the model (for example on dapi and lipid chanels).  
   ```console
    python train.py --coco_json_path 'jsons/coco_all_labels.json' --train_images_path 'train_images/' --target_channel_names 'dapi,lipid' --validate True 
    ```  
    After that in the repo folder there will appear directories /train_masks, /val_masks, /val_images and model file with default name 'unet'
5) To make the markup of the images in some folder (for example in /val_images) do the following (target_channel_names ORDER MATTERS!)
   ```console
    python markup.py --model_path './unet' --markup_images_path './val_images/' --target_channel_names 'dapi,lipid'
    ```
    After that the default folder './markups' will appear (name can be changed via adding --save_path argument)
6) To integrate the signal of each target chanel of each image w.r.t the corresponding mask we need apply.py file which is launched like this:
   ```console
   python apply.py --masks_directory_path 'markups' --images_path 'val_images' --target_channel_names 'dapi,lipid' 
   ``` 
   To integrate particular image instead of all of them in the folder use argument --interst_mask (for example 'dapi_0_0'):
   ```console  
   python apply.py --masks_directory_path 'markups' --images_path 'val_images' --target_channel_names 'dapi,lipid' --interst_mask 'dapi_0_0.tif'
   ```
   It will save default named json file 'channels_integrated_signal.json' with the following structure
   {"dapi": {"dapi_0_0.tif": 1098415}, "lipid": {"lipid_0_0.tif": 613816}}
