from torch.utils.data import Dataset 


class ImageDataset(Dataset):

    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
       
    def __getitem__(self, index):
        return self.data[index], self.labels[index]
    
    def __len__(self):
        return len(self.data)
 
    def __repr__(self):
        return f"<ImageDataset: {len(self)} samples, labels: {set(self.labels)}>"


    
