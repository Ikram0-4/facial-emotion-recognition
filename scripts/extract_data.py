"""Creation of the dataframe using FER-2013"""

import numpy as np 
import torch
import torch.nn as nn
 
import pandas as pd
import glob
import os 
from typing import List
# from keras.preprocessing.image import load_img
# from keras.preprocessing.image import img_to_array
# from keras.applications.vgg16 import preprocess_input
# from keras.applications.vgg16 import decode_predictions
# from keras.applications.vgg16 import VGG16
from torch.utils.data import DataLoader, TensorDataset #miniBatch pour traiter petit 


import torchvision
from torchvision.io import read_image
import torchvision.transforms as T







input_tensor = torch.randn(28709 , 3, 48, 48)

conv_layer = nn.Conv2d(
    in_channels=3,     # nombre de canaux en entrée (ex : 3 pour une image RGB)
    out_channels=48,   # nombre de filtres (ce que tu veux produire)
    kernel_size=48      # taille du filtre (ex : 3x3)
)

# dataset = 

# DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
#            batch_sampler=None, num_workers=0, collate_fn=None,
#            pin_memory=False, drop_last=False, timeout=0,
#            worker_init_fn=None, *, prefetch_factor=2,
#            persistent_workers=False)

# output = conv_layer(input_tensor)


def get_files(files):


    list_emotions = []
    images = []
    list_files = []
    for file_path in files:
        if os.path.isfile(file_path):
            list_files.append(file_path)
            emotions = os.path.basename(os.path.dirname(file_path))
            filename = os.path.basename(file_path)
            images.append(filename)
            list_emotions.append(emotions)
    # print(list_files)
    return list_emotions, images, list_files



def convert_image_into_vectors(list_files):
    list_loaded_images = []
    number_file = 1
    # print(list_files)
    for file in list_files:
        pic = torchvision.io.read_image(file)
        # convert this torch tensor to PIL image 
        PIL_img = T.ToPILImage()(pic)
        list_loaded_images.append(PIL_img)
        if number_file == 1:
            PIL_img.show()

        number_file += 1
        print(f"{number_file} / {len(list_files)}")
    
    return list_loaded_images


def create_dataframe(liste_emotions: List[str], images : List[str]) -> pd.DataFrame:
    
    df = pd.DataFrame({'trueLabel': liste_emotions, 'filename': images})
    one_hot = pd.get_dummies(df['trueLabel'])
    df_final = pd.concat([df, one_hot], axis=1)
    print(df_final.head())
    return df_final



def main():
    
    files = glob.glob("./data/train/**/*jpg", recursive=True)
    liste_emotions , liste_images, list_files = get_files(files)
    convert_image_into_vectors(list_files)


    # create_dataframe(liste_emotions, liste_images)


if __name__ == "__main__":
    main()