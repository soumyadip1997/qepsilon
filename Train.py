import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from importdatageo import data1,Train_Dataset,collate_fn_padd_train,Val_Dataset,collate_fn_padd_val
import numpy as np
import pandas as pd
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
torch.set_default_dtype(torch.float32)
import argparse
from torchsampler import ImbalancedDatasetSampler
from torch_geometric.nn import GCNConv,GraphConv
import csv
import glob
from lightning.pytorch.accelerators import find_usable_cuda_devices

from torch.utils.data import DataLoader
class GNN1(pl.LightningModule):
    def __init__(self,loss_type=1,lr=0.001,batch_size=1):
        super().__init__()
        self.loss_type=loss_type
        self.batch_size=batch_size
        self.lr=lr
        self.neigh_network_atom1 = GCNConv(12,1024)
        self.GCN_network_atom1=GraphConv(12,1024,"mean")
        self.GCN_network_residue1=GraphConv(1024,1024,"mean")
        self.neigh_network_atom2 = GCNConv(1024,512)
        self.GCN_network_atom2=GraphConv(1024,512,"mean")
        self.GCN_network_residue2=GraphConv(1024,512,"mean")
        self.neigh_network_atom3 = GCNConv(512,256)
        self.GCN_network_atom3=GraphConv(512,256,"mean")
        self.GCN_network_residue3=GraphConv(512,256,"mean")
        self.neigh_network_atom4 = GCNConv(256,128)
        self.GCN_network_atom4=GraphConv(256,128,"mean")
        self.GCN_network_residue4=GraphConv(256,128,"mean")
        self.drop1_atom=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop1_residue=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop2_atom=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop2_residue=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop3_atom=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop3_residue=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop4_atom=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop4_residue=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop_dense=torch.nn.Dropout(p=0.1, inplace=False)
        self.drop4=torch.nn.Dropout(p=0.1, inplace=False)
        self.relu=nn.ReLU()
        self.batchnorm_atom1=torch.nn.BatchNorm1d(1024)
        self.batchnorm_residue1=torch.nn.BatchNorm1d(1024)
        self.batchnorm_atom2=torch.nn.BatchNorm1d(512)
        self.batchnorm_residue2=torch.nn.BatchNorm1d(512)
        self.batchnorm_atom3=torch.nn.BatchNorm1d(256)
        self.batchnorm_residue3=torch.nn.BatchNorm1d(256)
        self.batchnorm_atom4=torch.nn.BatchNorm1d(128)
        self.batchnorm_residue4=torch.nn.BatchNorm1d(128)
        self.sigmoid1=nn.Sigmoid()
        self.dense=nn.Linear(256,1)
        #self.max_pool1=nn.MaxPool1d(10, stride=10)
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    def helper(self,atom,same,diff,res,res_neigh,GCN_network_atom_same,GCN_network_neigh_diff,GCN_network_residue,relu,batchnorm_atom,batchnorm_res,drop_atom,drop_res):
        atom_feat=GCN_network_atom_same(atom,same)
        neigh_feat=GCN_network_neigh_diff(atom,diff)
        final_atom=drop_atom(batchnorm_atom(relu(atom_feat+neigh_feat)))
        final_res=drop_res(batchnorm_res(relu(GCN_network_residue(res,res_neigh))))
        return final_atom,final_res 
    def training_step(self, train_batch, batch_idx):
        
        res,same_res_atom,diff_res_atom,atom,res_neigh,gdtts,req_eps,zeros,res_no,res_len = train_batch[0]
        atom_feat,res_feat=self.helper(atom,same_res_atom,diff_res_atom,res,res_neigh,self.GCN_network_atom1,self.neigh_network_atom1,self.GCN_network_residue1,self.relu,self.batchnorm_atom1,self.batchnorm_residue1,self.drop1_atom,self.drop1_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom2,self.neigh_network_atom2,self.GCN_network_residue2,self.relu,self.batchnorm_atom2,self.batchnorm_residue2,self.drop2_atom,self.drop2_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom3,self.neigh_network_atom3,self.GCN_network_residue3,self.relu,self.batchnorm_atom3,self.batchnorm_residue3,self.drop3_atom,self.drop3_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom4,self.neigh_network_atom4,self.GCN_network_residue4,self.relu,self.batchnorm_atom4,self.batchnorm_residue4,self.drop4_atom,self.drop4_residue)
        start=0
        mid=0
        local_scores=res_feat.clone()
        for i in range(len(res_no)):
            mid=start+res_no[i]
            local_scores[i]=torch.amax(atom_feat[start:mid],axis=0)
            start=mid
        res_feat=torch.concat((res_feat,local_scores),axis=1)    
        final_score=self.sigmoid1(self.dense(res_feat))
        #final_score=torch.mean(final_score,axis=0)
        final_score=torch.sum(final_score,axis=0,keepdim=True)
 
        final_score = torch.div(final_score,res_len) 
        if self.loss_type==1:
            loss =F.l1_loss(final_score.flatten(),gdtts.flatten())-req_eps
            #print(loss)
            if loss.item()<0:
                loss=zeros
        else:
            loss=F.l1_loss(final_score.flatten(),gdtts.flatten())
        return loss

    def validation_step(self, val_batch, batch_idx):
        res,same_res_atom,diff_res_atom,atom,res_neigh,gdtts,res_no,res_len = val_batch[0]
        atom_feat,res_feat=self.helper(atom,same_res_atom,diff_res_atom,res,res_neigh,self.GCN_network_atom1,self.neigh_network_atom1,self.GCN_network_residue1,self.relu,self.batchnorm_atom1,self.batchnorm_residue1,self.drop1_atom,self.drop1_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom2,self.neigh_network_atom2,self.GCN_network_residue2,self.relu,self.batchnorm_atom2,self.batchnorm_residue2,self.drop2_atom,self.drop2_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom3,self.neigh_network_atom3,self.GCN_network_residue3,self.relu,self.batchnorm_atom3,self.batchnorm_residue3,self.drop3_atom,self.drop3_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom4,self.neigh_network_atom4,self.GCN_network_residue4,self.relu,self.batchnorm_atom4,self.batchnorm_residue4,self.drop4_atom,self.drop4_residue)
        start=0
        mid=0
        local_scores=res_feat.clone()
        for i in range(len(res_no)):
            mid=start+res_no[i]
            local_scores[i]=torch.amax(atom_feat[start:mid],axis=0)
            start=mid
        res_feat=torch.concat((res_feat,local_scores),axis=1)    
        final_score=self.sigmoid1(self.dense(res_feat))
        #final_score=torch.mean(final_score,axis=0)
        final_score=torch.sum(final_score,axis=0,keepdim=True)
        final_score = torch.div(final_score,res_len) 
        loss=F.l1_loss(final_score.flatten(),gdtts.flatten())
        self.log('val_loss', loss,batch_size=self.batch_size)


# data
if  __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Lightening QA')
    parser.add_argument('--batch-size', type=int, default=70, metavar='N',
                        help='input batch size for training (default: 70)')
 
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of epochs to train (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--workers',type=int , default=12,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--save-model', default="Models/",
                        help='For Saving the current Model')
    parser.add_argument('--load-model', default="Models/",
                        help='For Loading the L1-Model')
    parser.add_argument('--devices',type=int, default=1 ,
                        help='Number of gpu devices')
    parser.add_argument('--nodes',type=int, default=1 ,
                        help='Number of nodes')
    parser.add_argument('--loss-type', type=int, default=0, metavar='LT',
                        help='loss type (Default L1-type)')
    parser.add_argument('--train-set', type=str, default="Train.npy", metavar='TR',
                        help='Train File')
    parser.add_argument('--val-set', type=str, default="Val.npy", metavar='VS',
                        help='Val set')
    parser.add_argument('--test-set', type=str, default="Test_CASP13_new.npy", metavar='TS',
                        help='Test set')
 

    parser.add_argument('--gdtts',type=str, default="Features/GDT_TS/gdtts_" ,
                        help='path to gdtts')
    parser.add_argument('--atom-one-hot',type=str, default="Features/ATOM/atom_one_hot_" ,
                        help='path to one hot atom encoding')
    parser.add_argument('--same-res-atom-neigh',type=str, default="Features/GRAPHNEIGH/Same_Res_Index_" ,
                        help='path to same residue atom neighbours')
    parser.add_argument('--diff-res-atom-neigh',type=str, default="Features/GRAPHNEIGH/Diff_Res_Index_" ,
                        help='path to diff residue atom neighbours')
    parser.add_argument('--res-neigh',type=str, default="Features/GRAPHNEIGH/Residue_Neigh_" ,
                        help='path to residue neighbour')
    parser.add_argument('--path-res-trans',type=str, default="Features/TRANS/Trans_" ,
                        help='path to transformer feature')
    parser.add_argument('--res-no',type=str, default="Features/Atomfreq/atomfreq_" ,
                        help='path to residue number')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    label_file_train=args.train_set
    label_file_val=args.val_set
    label_file_test=args.test_set
    gdtts=args.gdtts
    
    same_res_atom_neigh=args.same_res_atom_neigh
    diff_res_atom_neigh=args.diff_res_atom_neigh
    workers=args.workers
    batch_size=args.batch_size
    lr=args.lr
    res_neigh=args.res_neigh
    path_res_trans=args.path_res_trans
    atom_one_hot=args.atom_one_hot
    res_no=args.res_no


    temp=data1(path_res_trans,same_res_atom_neigh,diff_res_atom_neigh,atom_one_hot,res_neigh,gdtts,batch_size,workers,label_file_train,label_file_val,label_file_test,res_no)
    TD=Train_Dataset(temp)
    VD=Val_Dataset(temp)



    torch.set_float32_matmul_precision('high')

    

    train_loader = DataLoader(TD, batch_size=1,shuffle=False,num_workers=int(temp.workers),sampler=ImbalancedDatasetSampler(TD,TD.labels),collate_fn=collate_fn_padd_train)
    val_loader = DataLoader(VD, batch_size=1,shuffle=False,num_workers=int(temp.workers),collate_fn=collate_fn_padd_val)
    # model
    if args.loss_type==0:
        model = GNN1(0,args.lr,batch_size)#.load_from_checkpoint(path1)
        model.eval()
    else:
        path1=glob.glob(args.load_model)[0]

        model = GNN1(1,args.lr,batch_size).load_from_checkpoint(path1)
        model.loss_type=1
        model.eval()
    # training
    trainer = pl.Trainer(callbacks=[ModelCheckpoint(dirpath=args.save_model, save_top_k=1, monitor="val_loss")],min_epochs=3,accelerator="gpu",devices=find_usable_cuda_devices(args.devices), max_epochs=args.epochs,num_nodes=args.nodes,auto_select_gpus=True,accumulate_grad_batches=batch_size)#,limit_train_batches=70)  #,resume_from_checkpoint=path1)
    trainer.fit(model, train_loader,val_loader)

