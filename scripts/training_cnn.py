import torch
import torch.nn as nn
from extract_data import create_dataframe






def training_cnn():
    
    input_tensor = torch.randn(1, 16, 48, 48)

    conv_layer = nn.Conv2d(
        in_channels=3,     # nombre de canaux en entrée (ex : 3 pour une image RGB)
        out_channels=16,   # nombre de filtres (ce que tu veux produire pour caractèriser les images)
        kernel_size=3,
        padding=1     # taille du filtre (ex : 3x3)
    )

    output = conv_layer(input_tensor)
    return output

    # dataset = 

    # DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
    #            batch_sampler=None, num_workers=0, collate_fn=None,
    #            pin_memory=False, drop_last=False, timeout=0,
    #            worker_init_fn=None, *, prefetch_factor=2,
    #            persistent_workers=False)

print(training_cnn())