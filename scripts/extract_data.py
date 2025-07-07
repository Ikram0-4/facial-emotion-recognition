"""Creation of the dataframe using FER-2013 and creating a TensorDataset"""

import pandas as pd
import glob
import os 
from typing import List, Tuple
import torchvision
from torchvision.io import read_image
import torchvision.transforms as T
from torch.utils.data import DataLoader 
from models import ImageDataset

 


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
    # print(list_files)
    for file in list_files:
        pic = torchvision.io.read_image(file)
        PIL_img = T.ToPILImage()(pic) ##  to PIL image 
        to_tensor = T.ToTensor() ## convert a PIL image to an array
        image_tensor = to_tensor(PIL_img)
        list_loaded_images.append(image_tensor)
    # print(list_loaded_images)
    return list_loaded_images


def create_dataframe(liste_emotions: List[str], image_path : List[str], image : List[str] ) -> pd.DataFrame:
    
    df = pd.DataFrame({'trueLabel': liste_emotions, 'filename': image_path, 'Tensor':image })
    one_hot = pd.get_dummies(df['trueLabel'])
    df_final = pd.concat([df, one_hot], axis=1)
    #print(df_final.head())
    return df_final



def create_TensorDataset(dataframe : pd.DataFrame):
    
    label_map = {
    "angry": 0,
    "disgust": 1, 
    "fear": 2,
    "happy": 3,
    "neutral": 4,
    "sad": 5,
    "surprise": 6
}

    images = dataframe['Tensor']
    labels = dataframe['trueLabel'].map(label_map)
    tensor_data = ImageDataset(images, labels)
    return tensor_data
    

def create_batch(dataset):
     
    return DataLoader(dataset, batch_size=32, shuffle=True)

def main():
    
    files = glob.glob("../data/train/**/*jpg", recursive=True)
    liste_emotions , liste_files, list_image = get_files(files)
    images = convert_image_into_vectors(list_image)
    dataframe = create_dataframe(liste_emotions, liste_files, images)
    data_tensors = create_TensorDataset(dataframe)
    batches = create_batch(data_tensors)


    for i, (images_batch, labels_batch) in enumerate(batches):
        print(f"Batch {i + 1}")
        print(f"Images shape: {images_batch.shape}")
        print(f"Labels: {labels_batch}")
        break
 
    
if __name__ == "__main__":
    main()