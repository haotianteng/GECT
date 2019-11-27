import torch.utils.data as data
import torch
import pandas as pd
import numpy as np
from torchvision import transforms

FEATURE_ENTRY = 'rpkm'
LABEL_ENTRY = 'labels'
FEATURE_DTYPE = np.float32
EPSILON = 1e-6
class dataset(data.Dataset):
    """
    scRNA sequencing dataset, contain gene expression and cell type.
    """
    def __init__(self, store_file, transform=None):
        """
        Args:
            data_file (string): Path to the pandas hdf5 data storage file(contain
                      both feature and labels).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        store_f = pd.HDFStore(store_file)
        self.feature = np.asarray(store_f[FEATURE_ENTRY],dtype = FEATURE_DTYPE)
        self.transform = transform
        labels = store_f[LABEL_ENTRY]
        self.label_tags = np.unique(labels)
        for tag_idx,tag in enumerate(self.label_tags):
            labels[labels==tag] = tag_idx
        self.sample_n = len(labels)
        self.labels = np.reshape(np.asarray(labels,dtype = np.int32),newshape = (self.sample_n,1))
        self.gene_mean = np.mean(self.feature,axis = 0)
        self.gene_std = np.std(self.feature,axis = 0) + EPSILON
        store_f.close()
    def __len__(self):
        return self.sample_n

    def __getitem__(self, idx):
        item = {'feature': self.feature[idx],
                'label': self.labels[idx],
                'feature_mean':self.gene_mean,
                'feature_std':self.gene_std}
        if self.transform:
            item = self.transform(item)
        return item
    
class MeanNormalization(object):
    """Nomalization method used to the gene expression level.
    """
    def __call__(self,item):
        feature = item['feature'] - item['feature_mean']
        feature = feature/item['feature_std']
        return {'feature':feature,
                'label':item['label']}
        
class Crop(object):
    """Crop some feature
    """
    pass

class ToTensor(object):
    def __call__(self, item):
        return {'feature':torch.from_numpy(item['feature']),
                'label':torch.from_numpy(item['label'])}

class Embedding(object):
    """
    """
    def __init__(self,embedding_matrix):
        self.embedding = embedding_matrix
    def __call__(self,sample):
        feature = sample['feature']
        ##TODO implemented the embedding
  
class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dataloader, device = None):
        self.dataloader = dataloader
        if device is None:
            device = self.get_default_device()
        else:
            device = torch.device(device)
        self.device = device
    
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dataloader:
            yield self._to_device(b, self.device)
    
    def __len__(self):
        """Number of batches"""
        return len(self.dataloader)
    
    def _to_device(self,data,device):
        if isinstance(data, (list,tuple)):
            return [self._to_device(x,device) for x in data]
        if isinstance(data, (dict)):
            temp_dict = {}
            for key in data.keys():
                temp_dict[key] = self._to_device(data[key],device)
            return temp_dict
        return data.to(device, non_blocking=True)
    
    def get_default_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

if __name__ == "__main__":
    root_dir = '/home/heavens/CMU/GECT/data/'
    train_dat = "train_data.h5"
    test_dat = "test_data.h5"
#    d1 = dataset(root_dir,transform=transforms.Compose([DeNoise((200,1200)),
#                                                        WhiteNoise(200,0.1),
#                                                        Crop(30000),
#                                                        TransferProb(5),
#                                                        ToTensor()]))
    d1 = dataset(root_dir+train_dat,transform=transforms.Compose([MeanNormalization(),
                                                            ToTensor()]))
    dataloader = data.DataLoader(d1,batch_size=10,shuffle=True,num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        feature = sample_batched['feature']
        label = sample_batched['label']
        print(feature.shape)
        print(label.shape)
        if i_batch>10:
            break

    

