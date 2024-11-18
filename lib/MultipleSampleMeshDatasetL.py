
import numpy as np
import torch 
from torch.utils.data import Dataset
import scipy.stats

import os
import hcp_utils as hcp

class MultipleSampleMeshDatasetL(Dataset):
    def __init__(self, subj_ids, rsfc_dir, contrast_dir, num_samples=8, cache=True, switch_dims=False, yield32k=True, zscore_tasks=True):
        self.rsfc_dir = rsfc_dir
        self.contrast_dir = contrast_dir
        self.subj_ids = subj_ids
        self.num_samples = num_samples
        self.greyl = hcp.vertex_info["grayl"]
        self.rsfcd = {}
        self.taskd = {}
        self.yield32k= yield32k;
        self.switch_dims=switch_dims;
        self.cache= cache;
        self.zscore_tasks= zscore_tasks;
        
    def get_specific_item(self, index, sample_id):
        subj = self.subj_ids[index]
        key = subj + str(sample_id); 
        rsfc = None
        taskc = None
        
        if key in self.rsfcd.keys() and self.cache:
          rsfc = self.rsfcd[key]
        else:
          rsfc_file = os.path.join(self.rsfc_dir, "L_%s_sample%d_rsfc.npy" % (subj, sample_id))
          subj_rsfc_data = np.load(rsfc_file)
          
          rsfc = subj_rsfc_data
          if self.yield32k:
            rsfc = np.zeros((subj_rsfc_data.shape[0], 32492))
            rsfc[:, self.greyl] = subj_rsfc_data; 
        
          if self.switch_dims:
            rsfc = np.moveaxis(rsfc, -2,-1)
            
          if self.cache: self.rsfcd[key] = rsfc
        
        if subj in self.taskd.keys() and self.cache:
          taskc = self.taskd[subj]
        else:
          subj_task_data = np.load(os.path.join(self.contrast_dir, "%s_joint_L_task_contrasts.npy" % subj))
          
          taskc = subj_task_data
          if self.yield32k:
            taskc = np.zeros((subj_task_data.shape[0], 32492))
            taskc[:, self.greyl] = subj_task_data; 
          
          if self.zscore_tasks:
            taskc = scipy.stats.zscore(taskc, axis=-1)
            
          if self.switch_dims:
            taskc = np.moveaxis(taskc, -2,-1)
        
          if self.cache: self.taskd[subj] = taskc
        
        #return torch.cuda.FloatTensor(subj_rsfc_data) , torch.cuda.FloatTensor(subj_task_data)
        return torch.FloatTensor(rsfc) , torch.FloatTensor(taskc)

    def __getitem__(self, index):
        sample_id = np.random.randint(0, self.num_samples)        
        return self.get_specific_item(index, sample_id)

    def __len__(self):
        return len(self.subj_ids)
