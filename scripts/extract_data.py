"""Creation of the dataframe using FER-2013 and creating a TensorDataset"""

import pandas as pd
import glob
import os 
from typing import List, Tuple
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
from torch.utils.data import DataLoader, TensorDataset  


def get_files(files : List[str]) -> Tuple[List[str], List[str], List[str]]:


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



def convert_image_into_vectors(list_files: List[str])-> List[str]:
  
    list_loaded_images = []
    # number_file = 1
    # print(list_files)
    for file in list_files:
        pic = torchvision.io.read_image(file)
        PIL_img = T.ToPILImage()(pic) ##  to PIL image 
        to_tensor = T.ToTensor() ## convert a PIL image to an array
        image_tensor = to_tensor(PIL_img)
        list_loaded_images.append(image_tensor)
        # if number_file == 1:
        #     PIL_img.show()
        # number_file += 1
        # print(f"{number_file} / {len(list_files)}")
    # print(list_loaded_images)
    return list_loaded_images


def create_dataframe(liste_emotions: List[str], image_path : List[str], image : List[str] ) -> pd.DataFrame:
    
    df = pd.DataFrame({'trueLabel': liste_emotions, 'filename': image_path, 'Tensor':image })
    one_hot = pd.get_dummies(df['trueLabel'])
    df_final = pd.concat([df, one_hot], axis=1)
    print(df_final.head())
    return df_final


def create_TensorDataset():
    ...
    


def main():
    
    files = glob.glob("./data/train/**/*jpg", recursive=True)
    liste_emotions , liste_files, list_image = get_files(files)
    images = convert_image_into_vectors(list_image)
    create_dataframe(liste_emotions, liste_files, images)


if __name__ == "__main__":
    main()