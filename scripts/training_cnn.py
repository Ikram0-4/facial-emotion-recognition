import torch
import torch.nn as nn
from extract_data import create_dataframe, create_TensorDataset
from extract_data import get_train_batches
from torch.utils.data import DataLoader 



def training_cnn(batches: DataLoader):
    
    conv_layer = nn.Conv2d(
        in_channels=3,
        out_channels=16,
        kernel_size=3,
        padding=1
    )

    for batch in batches:
        images, labels = batch

        ## duplicate 1 image canal in 3 canals
        images = images.repeat(1, 3, 1, 1)
        output = conv_layer(images)
        
        print(f"Input: {images.shape}, Output: {output.shape}")
        break 

    return output
    
      

    # dataset = 

    # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
    #            batch_sampler=None, num_workers=0, collate_fn=None,
    #            pin_memory=False, drop_last=False, timeout=0,
    #            worker_init_fn=None, *, prefetch_factor=2,
    #            persistent_workers=False)

training_cnn(get_train_batches())
