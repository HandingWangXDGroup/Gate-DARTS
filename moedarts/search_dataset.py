
import torch
import torch.utils.data as data

class SearchDataset(data.Dataset):
    def __init__(self,data,indices,split) -> None:
        super().__init__()
        self.data = data
        self.train_split = indices[:split]
        self.val_split = indices[split:]
        len_train = len(self.train_split)
        len_val = len(self.val_split)
        self.length = min(len_train,len_val)
        if len_train!=len_val:
            self.train_split = self.train_split[:self.length]
            self.val_split = self.val_split[:self.length]
    
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        assert index>=0 and index < self.length, "invalid index = {}".format(index)
        train_index = self.train_split[index]
        val_index = self.val_split[index]
        train_image, train_label = self.data[train_index]
        val_image,  val_label = self.data[val_index]
        return train_image, train_label, val_image, val_label