import random
import torch
from torch.utils.data import Dataset
from torchsampler import ImbalancedDatasetSampler
import torchvision.transforms as transforms
from torch.utils.data import sampler
import six
import sys
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#torch.multiprocessing.set_sharing_strategy('file_system')
def make_weights_for_balanced_classes(interactions, nclasses=10):                        
    count = [0] * nclasses                                                      
    for item in interactions:     
        if item=="pos":
                                                    
            count[1] += 1     
        else:
            count[0]+=1                                                
    weight_per_class = [0.] * nclasses                                      
    N = float(sum(count))                                                   
    for i in range(nclasses):                                                   
        weight_per_class[i] = N/float(count[i])                                 
    weight = [0] * len(interactions)                                              
    for idx, val in enumerate(interactions):
        if val=="pos":                                          
            weight[idx] = weight_per_class[1]
        else:
            weight[idx]=weight_per_class[0]                                  
    return weight         
class Train_Dataset(Dataset):
    def __init__(self, opt=None):
        self.train_dataset = opt.train_dataset
        self.target_name=self.train_dataset[:,1]
        self.decoy_name=self.train_dataset[:,2]
        self.test_dataset=opt.test_dataset
        self.trans=opt.trans
        self.same_res_atom_neigh=opt.same_res_atom_neigh
        self.diff_res_atom_neigh=opt.diff_res_atom_neigh
        self.atom_one_hot=opt.atom_one_hot
        self.res_neigh=opt.res_neigh
        self.gdtts=opt.gdtts
        self.gdtha=opt.gdtha
        self.res_no=opt.res_no
        self.tmscore=opt.tmscore
        self.gcad=opt.gcad
        self.lcad=opt.lcad
        self.num_classes=10
        self.count=[0]*10
        self.eps=[0.45,0.4,0.35,0.3,0.25,0.2,0.15,0.1,0.01,0.01,0.01]
        self.class_idx_to_sample_ids={i:[] for i in range(self.num_classes)}
        self.labels=[]
        for i in range(len(self.train_dataset)):
                score=round(float(self.train_dataset[i][3])*10)
                #print(i,score)    
                if score==10:
                    score=9
                self.labels.append(score)
                self.class_idx_to_sample_ids[score].append(i)
        max_value=-189
    def __len__(self):
        return len(self.train_dataset)
    def __getitem__(self,index1):
            sample_idx=index1
            res_feat=torch.tensor(np.load(self.trans+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy")[0],dtype=torch.float32, requires_grad=True)
            same_res_atom=torch.tensor(np.load(self.same_res_atom_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            diff_res_atom=torch.tensor(np.load(self.diff_res_atom_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            atom_one_hot=torch.tensor(np.load(self.atom_one_hot+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.float32, requires_grad=True)
            res_neigh=torch.tensor(np.load(self.res_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            gdtts=torch.tensor(np.load(self.gdtts+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"), requires_grad=True)
            gdtha=torch.tensor(np.load(self.gdtha+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"), requires_grad=True)
            tmscore=torch.tensor(np.load(self.tmscore+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"), requires_grad=True)
            gcad=torch.tensor(np.load(self.gcad+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"))
            lcad=np.load(self.lcad+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy")
            lcad=torch.tensor(np.nan_to_num(lcad)) 
            num_res=len(res_feat)
            req_eps=torch.tensor(self.eps[round(gdtts.item()*10)],dtype=torch.float32, requires_grad=True)
            zeros=torch.tensor([0],dtype=torch.float32, requires_grad=True)
            res_no=torch.tensor(np.load(self.res_no+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            return res_feat,same_res_atom,diff_res_atom,atom_one_hot,res_neigh,gdtts,gdtha,tmscore,gcad,lcad,req_eps ,zeros,res_no
class Val_Dataset(Dataset):
    def __init__(self, opt=None):
        self.val_dataset=opt.val_dataset
        self.target_name=self.val_dataset[:,1]
        self.decoy_name=self.val_dataset[:,2]
        self.trans=opt.trans
        self.same_res_atom_neigh=opt.same_res_atom_neigh
        self.diff_res_atom_neigh=opt.diff_res_atom_neigh
        self.atom_one_hot=opt.atom_one_hot
        self.res_neigh=opt.res_neigh
        self.gdtts=opt.gdtts
        self.gdtha=opt.gdtha
        self.tmscore=opt.tmscore
        self.gcad=opt.gcad
        self.lcad=opt.lcad
        self.res_no=opt.res_no
    def __len__(self):
        return len(self.val_dataset)
    def __getitem__(self,index1):
            sample_idx=index1 
            res_feat=torch.tensor(np.load(self.trans+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy")[0],dtype=torch.float32)
            same_res_atom=torch.tensor(np.load(self.same_res_atom_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            diff_res_atom=torch.tensor(np.load(self.diff_res_atom_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            atom_one_hot=torch.tensor(np.load(self.atom_one_hot+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.float32)
            res_neigh=torch.tensor(np.load(self.res_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            gdtts=torch.tensor(np.load(self.gdtts+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"))
            res_no=torch.tensor(np.load(self.res_no+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"))
            return res_feat,same_res_atom,diff_res_atom,atom_one_hot,res_neigh,gdtts,res_no#gdtha,tmscore,gcad,lcad,name
class Test_Dataset(Dataset):
    def __init__(self, opt=None):
        self.train_dataset = opt.train_dataset
        self.test_dataset=opt.test_dataset
        self.target_name=self.test_dataset[:,1]
        self.decoy_name=self.test_dataset[:,2]
        self.trans=opt.trans
        self.same_res_atom_neigh=opt.same_res_atom_neigh
        self.diff_res_atom_neigh=opt.diff_res_atom_neigh
        self.atom_one_hot=opt.atom_one_hot
        self.res_neigh=opt.res_neigh
        self.gdtts=opt.gdtts
        #self.gdtha=opt.gdtha
        #self.tmscore=opt.tmscore
        #self.gcad=opt.gcad
        #self.lcad=opt.lcad
        self.res_no=opt.res_no
    def __len__(self):
        return len(self.test_dataset)
    def __getitem__(self,index1):
        #try:
            sample_idx=index1 
            res_feat=torch.tensor(np.load(self.trans+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy")[0],dtype=torch.float32)
            same_res_atom=torch.tensor(np.load(self.same_res_atom_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            res_no=torch.tensor(np.load(self.res_no+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"))
            diff_res_atom=torch.tensor(np.load(self.diff_res_atom_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            atom_one_hot=torch.tensor(np.load(self.atom_one_hot+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.float32)
            res_neigh=torch.tensor(np.load(self.res_neigh+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"),dtype=torch.long)
            gdtts=torch.tensor(np.load(self.gdtts+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"))
            #gdtha=torch.tensor(np.load(self.gdtha+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"))
            #tmscore=torch.tensor(np.load(self.tmscore+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"))
            #gcad=torch.tensor(np.load(self.gcad+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy"))
            #lcad=np.load(self.lcad+self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx]+".npy")
            #lcad=torch.tensor(np.nan_to_num(lcad)) 
            name=self.target_name[sample_idx]+"_"+self.decoy_name[sample_idx] 
            return res_feat,same_res_atom,diff_res_atom,atom_one_hot,res_neigh,gdtts,name,res_no#gdtha,tmscore,gcad,lcad,name
def collate_fn_padd_train(batch):
    return batch
def collate_fn_padd_val(batch):
    return batch
def collate_fn_padd_test(batch):
    return batch
def get_dataloader_train():
    train_dataset = PDB_Train()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1,shuffle=True,num_workers=0)
    return train_loader

class data1:
    def __init__(self,transformer,same_res_atom_neigh,diff_res_atom_neigh,atom_one_hot,res_neigh,gdtts,batch_size,workers,label_file_train,label_file_val,label_file_test,res_no):
        self.train_dataset=np.load(label_file_train)
        self.val_dataset=np.load(label_file_val)
        self.test_dataset=np.load(label_file_test)
        self.trans=transformer
        self.same_res_atom_neigh=same_res_atom_neigh
        self.diff_res_atom_neigh=diff_res_atom_neigh
        self.atom_one_hot=atom_one_hot
        self.res_neigh=res_neigh
        self.batchSize=batch_size
        self.workers=workers
        self.gdtts=gdtts
        self.res_no=res_no
if  __name__ == "__main__":
    label_file="Final_Decoy_List_GEO.npy"
    gdtts="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GDT_TS/gdtts_"
    gdtha="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GDT_HA/gdtha_"
    gcad="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/CAD/globalcad_"
    lcad="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/CAD/localcad_"
    tmscore="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/TMscore/tmscore_"
    same_res_atom_neigh="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GRAPHNEIGH/Same_Res_Index_"
    diff_res_atom_neigh="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GRAPHNEIGH/Diff_Res_Index_"
    workers=10
    batch_size=1
    res_neigh="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GRAPHNEIGH/Residue_Neigh_"
    path_res_trans="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/TRANS/Trans_"
    atom_one_hot="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/ATOM/atom_one_hot_"
    train_data=["9","10","11","12"]
    val_data=None
    test_data=["13"]
    
    
    temp=data1(path_res_trans,same_res_atom_neigh,diff_res_atom_neigh,atom_one_hot,res_neigh,gdtts,gdtha,tmscore,gcad,lcad,batch_size,workers,train_data,val_data,test_data,label_file)
    TD=Train_Dataset(temp)
    train_loader = torch.utils.data.DataLoader(TD, batch_size=temp.batchSize,shuffle=False,num_workers=temp.workers,sampler=ImbalancedDatasetSampler(TD,TD.labels),collate_fn=collate_fn_padd_train)
    for i,j in enumerate(train_loader):
        print(i,j[5])
 
