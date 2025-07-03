"""Creation of the dataframe using FER-2013"""

import numpy as np 
from tensorflow import keras 
import pandas as pd
import glob
import os 
from typing import List

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

model = VGG16()


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
   
    return list_emotions, images, list_files


def convert_image_into_vectors(list_files):
    list_loaded_images = []
    number_file = 1
    for file in list_files:
        ## loading image with keras
        image_load = load_img(file, target_size=(48, 48))
        list_loaded_images.append(image_load)
        number_file += 1
        print(f"{number_file} / {len(list_files)}")
    return list_loaded_images


def create_dataframe(liste_emotions: List[str], images : List[str]) -> pd.DataFrame:
    
    df = pd.DataFrame({'trueLabel': liste_emotions, 'filename': images})
    one_hot = pd.get_dummies(df['trueLabel'])
    df_final = pd.concat([df, one_hot], axis=1)
    #print(df_final.head())
    return df_final



def main():
    
    files = glob.glob("../data/train/**/*jpg", recursive=True)
    liste_emotions , liste_images, list_files = get_files(files)
    convert_image_into_vectors(list_files)


    # create_dataframe(liste_emotions, liste_images)


if __name__ == "__main__":
    main()