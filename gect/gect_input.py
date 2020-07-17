import torch.utils.data as data
import torch
import pandas as pd
import numpy as np
from torchvision import transforms
import os

FEATURE_ENTRY = 'rpkm'
LABEL_ENTRY = 'labels'
FEATURE_DTYPE = np.float32
EPSILON = 1e-6
GENE_COL = np.arange(9,164)
HEADER = 0
class dataset(data.Dataset):
    """
    scRNA sequencing dataset, contain gene expression and cell type.
    """
    def __init__(self, 
                 store_file, 
                 transform=None):
        """
        Args:
            data_file (string): Path to the pandas hdf5 data storage file(contain
                      both feature and labels).
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if store_file.endswith('h5') or store_file.endswith('fast5'):
            store_f = pd.HDFStore(store_file,'r')
            self.feature = np.asarray(store_f[FEATURE_ENTRY],dtype = FEATURE_DTYPE)
            self.gene_n = self.feature.shape[1]
            self.transform = transform
            labels = store_f[LABEL_ENTRY]
            store_f.close()
        else:
            if store_file.endswith('.xlsx'):
                data_all = pd.read_excel(store_file,header = HEADER)
            elif store_file.endswith('.csv'):
                data_all = pd.read_csv(store_file,header = HEADER)
            self.feature = np.asarray(data_all.iloc[:,GENE_COL],dtype = FEATURE_DTYPE)
            self.gene_n = self.feature.shape[1]
            labels = np.asarray(data_all['Cell_class'])
            self.transform = transform
                    
        self.label_tags = np.unique(labels)
        for tag_idx,tag in enumerate(self.label_tags):
            labels[labels==tag] = tag_idx
        self.sample_n = len(labels)
        self.labels = np.reshape(np.asarray(labels,dtype = np.int64),newshape = (self.sample_n,1))
        self.gene_mean = np.mean(self.feature,axis = 0)
        self.gene_std = np.std(self.feature,axis = 0) + EPSILON
        self.cell_type_n = len(self.label_tags)
    def __len__(self):
        return self.sample_n

    def __getitem__(self, idx):
        item = {'feature': self.feature[idx],
                'label': self.labels[idx],
                'label_tags': self.label_tags,
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
                'label':item['label'],
                'label_tags':item['label_tags']}

class ToTags(object):
    """Transfer label using given tags.
    """
    def __init__(self,tags):
        self.tags = tags
    def __call__(self,sample):
        label = sample['label']
        label_tags = sample['label_tags']
        new_label = np.where(self.tags==label_tags[label[0]])[0]
        assert(len(new_label)==1)
        return {'feature':sample['feature'],
                'label':new_label,
                'label_tags':label_tags}
class OnehotEncoding(object):
    """Encoding the label with one-hot vector, unneccassory as PyTorch CELoss
    do the OneHotEncoding internally.
    """
    def __call__(self,sample):
        label_idx = sample['label']
        label = np.zeros(len(sample['label_tags']))
        label[label_idx] = 1
        return {'feature':sample['feature'],
                'label':label}

class Embedding(object):
    """Transfer the gene expression level into embedding vector
    """
    def __init__(self,embedding_matrix):
        self.embedding = embedding_matrix
    def __call__(self,sample):
        feature = sample['feature']
        feature = np.matmul(feature,self.embedding)
        return {'feature':feature,
                'label':sample['label']}
        
        
class ToTensor(object):
    def __call__(self, item):
        return {'feature':torch.from_numpy(item['feature']),
                'label':torch.from_numpy(item['label']).squeeze()}
  
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

def cell_list(y, cell_n):
    """Chooce cell_n cell types with most instances.
    """
    uniq,counts = np.unique(y,return_counts =True)
    sub_cell_list = uniq[np.argsort(counts)]
    sub_cell_list = sub_cell_list[-cell_n:]
    return sub_cell_list

def get_sub_data(x,y,sub_cell_list):
    sub_cell_choice = y==sub_cell_list[0]
    for t in sub_cell_list[1:]:
        sub_cell_choice = np.logical_or(sub_cell_choice,y==t)
    sub_x = x[sub_cell_choice]
    sub_label = y[sub_cell_choice]
    sub_y = np.copy(sub_label)
    for idx,label in enumerate(sub_cell_list):
        sub_y[sub_label==label] = idx
    return sub_x,sub_y
def transfer_label_tags(y,label_tags_from,label_tags_to):
    labels = np.unique(y)
    new_y = np.copy(y)
    for label in labels:
        new_y[y==label] = np.where(label_tags_to == label_tags_from[label])[0][0]
    return new_y

def load_embedding(model_folder):
    ckpt_file = os.path.join(model_folder,'checkpoint') 
    with open(ckpt_file,'r') as f:
        latest_ckpt = f.readline().strip().split(':')[1]
    state_dict = torch.load(os.path.join(model_folder,latest_ckpt))
    embedding_matrix = state_dict['linear1.weight'].detach().cpu().numpy()
    return embedding_matrix.transpose()


if __name__ == "__main__":
#    root_dir = '/home/heavens/CMU/GECT/data/'
#    train_dat = "train_data.h5"
#    test_dat = "test_data.h5"
#    d1 = dataset(root_dir,transform=transforms.Compose([DeNoise((200,1200)),
#                                                        WhiteNoise(200,0.1),
#                                                        Crop(30000),
#                                                        TransferProb(5),
#                                                        ToTensor()]))
    data_f = "/home/heavens/CMU/FISH_Clustering/MERFISH2018/naive.csv"
    d1 = dataset(data_f,transform=transforms.Compose([MeanNormalization(),
                                                            ToTensor()]))
    dataloader = data.DataLoader(d1,batch_size=10,shuffle=True,num_workers=4)
    for i_batch, sample_batched in enumerate(dataloader):
        feature = sample_batched['feature']
        label = sample_batched['label']
        print(feature.shape)
        print(label.shape)
        if i_batch>10:
            break

    

