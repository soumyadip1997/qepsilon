import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
from importdatageo import data1,Train_Dataset,collate_fn_padd_train,Test_Dataset,collate_fn_padd_test
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
from torch.utils.data import DataLoader
import scipy.stats as ss1
class GNN1(pl.LightningModule):
    def __init__(self,batch_size=1):
        super().__init__()
        self.File1=open("Result_CASP14_new.csv","w+")
        self.write1=csv.writer(self.File1)
        #self.source_atom_one_hot_1 = nn.Linear(12,1024)
        self.batch_size=1
        self.neigh_network_atom1 = GCNConv(12,1024)
        self.GCN_network_atom_same1=GraphConv(12,1024,"mean")
        self.GCN_network_residue1=GraphConv(1024,1024,"mean")
        self.neigh_network_atom2 = GCNConv(1024,512)
        self.GCN_network_atom_same2=GraphConv(1024,512,"mean")
        self.GCN_network_residue2=GraphConv(1024,512,"mean")
        self.neigh_network_atom3 = GCNConv(512,256)
        self.GCN_network_atom_same3=GraphConv(512,256,"mean")
        self.GCN_network_residue3=GraphConv(512,256,"mean")
        self.neigh_network_atom4 = GCNConv(256,128)
        self.GCN_network_atom_same4=GraphConv(256,128,"mean")
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
        self.list1=[[0,0]]
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer
    def helper(self,atom,same,diff,res,res_neigh,GCN_network_atom_same,GCN_network_neigh,GCN_network_residue,relu,batchnorm_atom,batchnorm_res,drop_atom,drop_res):
        atom_feat=GCN_network_atom_same(atom,sam70e)
        neigh_feat=GCN_network_neigh(atom,diff)
        final_atom=drop_atom(batchnorm_atom(relu(atom_feat+neigh_feat)))
        final_res=drop_res(batchnorm_res(relu(GCN_network_residue(res,res_neigh))))
        return final_atom,final_res 
    def training_step(self, train_batch, batch_idx):
        
        res,same_res_atom,diff_res_atom,atom,res_neigh,gdtts,gdtha,tmscore,gcad,lcad,num_res,req_eps = train_batch[0][0],train_batch[0][1],train_batch[0][2],train_batch[0][3],train_batch[0][4],train_batch[0][5],train_batch[0][6],train_batch[0][7],train_batch[0][8],train_batch[0][9],train_batch[0][10],train_batch[0][11]
        atom_feat,res_feat=self.helper(atom,same_res_atom,diff_res_atom,res,res_neigh,num_res,self.GCN_network_atom_same1,self.neigh_network_atom1,self.GCN_network_residue1,self.relu,self.batchnorm_atom1,self.batchnorm_residue1,self.drop1_atom,self.drop1_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,num_res,self.GCN_network_atom_same2,self.neigh_network_atom2,self.GCN_network_residue2,self.relu,self.batchnorm_atom2,self.batchnorm_residue2,self.drop2_atom,self.drop2_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,num_res,self.GCN_network_atom_same3,self.neigh_network_atom3,self.GCN_network_residue3,self.relu,self.batchnorm_atom3,self.batchnorm_residue3,self.drop3_atom,self.drop3_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,num_res,self.GCN_network_atom_same4,self.neigh_network_atom4,self.GCN_network_residue4,self.relu,self.batchnorm_atom4,self.batchnorm_residue4,self.drop4_atom,self.drop4_residue)
        local_scores=self.max_pool1(atom_feat.permute(1,0))
        local_scores=local_scores.permute(1,0)
        res_feat[:min(len(res_feat),len(local_scores))]=res_feat[:min(len(res_feat),len(local_scores))]+local_scores[:min(len(res_feat),len(local_scores))]
        final_score=self.sigmoid1(self.dense(res_feat))
        final_score=torch.mean(final_score,axis=0)
        loss =F.l1_loss(final_score.flatten(),gdtts.flatten())
        self.log('train_loss', loss)
        return loss
    def test_step(self, test_batch, batch_idx):
        res,same_res_atom,diff_res_atom,atom,res_neigh,gdtts,index,res_no = test_batch[0]#,train_batch[0][1],train_batch[0][2],train_batch[0][3],train_batch[0][4],train_batch[0][5],train_batch[0][6],train_batch[0][7],train_batch[0][8],train_batch[0][9],train_batch[0][10],train_batch[0][11]
        atom_feat,res_feat=self.helper(atom,same_res_atom,diff_res_atom,res,res_neigh,self.GCN_network_atom_same1,self.neigh_network_atom1,self.GCN_network_residue1,self.relu,self.batchnorm_atom1,self.batchnorm_residue1,self.drop1_atom,self.drop1_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom_same2,self.neigh_network_atom2,self.GCN_network_residue2,self.relu,self.batchnorm_atom2,self.batchnorm_residue2,self.drop2_atom,self.drop2_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom_same3,self.neigh_network_atom3,self.GCN_network_residue3,self.relu,self.batchnorm_atom3,self.batchnorm_residue3,self.drop3_atom,self.drop3_residue)
        atom_feat,res_feat=self.helper(atom_feat,same_res_atom,diff_res_atom,res_feat,res_neigh,self.GCN_network_atom_same4,self.neigh_network_atom4,self.GCN_network_residue4,self.relu,self.batchnorm_atom4,self.batchnorm_residue4,self.drop4_atom,self.drop4_residue)
        #local_scores=self.max_pool1(atom_feat.permute(1,0))
        #local_scores=local_scores.permute(1,0)
        #res_feat[:min(len(res_feat),len(local_scores))]=res_feat[:min(len(res_feat),len(local_scores))]+local_scores[:min(len(res_feat),len(local_scores))]
        start=0
        mid=0
        local_scores=res_feat.clone()
        for i in range(len(res_no)):
            mid=start+res_no[i]
            local_scores[i]=torch.amax(atom_feat[start:mid],axis=0)
            start=mid
        res_feat=torch.concat((res_feat,local_scores),axis=1) 
        final_score=self.sigmoid1(self.dense(res_feat))
        final_score=torch.mean(final_score,axis=0)
        loss =F.l1_loss(final_score.flatten(),gdtts.flatten())
        gdtts=gdtts.flatten()
        final_score=final_score.flatten()
        for i in range(len(final_score)):
            fields=[index,gdtts[i].item(),final_score[i].item()]
            self.write1.writerow(fields)
            self.list1.append([gdtts[i].item(),final_score[i].item()])
        list2=np.array(self.list1).reshape(-1,2)
        corr=ss1.pearsonr(list2[:,0],list2[:,1])[0]
        print(corr)
        self.log('loss',loss ,batch_size=self.batch_size,prog_bar=True)


# data
if  __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Lightening QA')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--test-batch-size', type=int, default=70, metavar='N',
                        help='input batch size for testing (default: 70)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--workers',type=int , default=0,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--model-path', default="lightning_logs/version_3182/checkpoints/epoch=9*.ckpt",
                        help='For Saving the current Model')
    parser.add_argument('--devices',type=int, default=1 ,
                        help='Number of gpu devices')
    parser.add_argument('--nodes',type=int, default=1 ,
                        help='Number of nodes')
    parser.add_argument('--test-set',type=str, default="Test_CASP13_new.npy" ,
                        help='path to Test Set')
    parser.add_argument('--train-set',type=str, default="Train.npy" ,
                        help='path to Train Set')
    parser.add_argument('--val-set',type=str, default="Val.npy" ,
                        help='path to Val Set')
    parser.add_argument('--gdtts',type=str, default="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GDT_TS/gdtts_" ,
                        help='path to gdtts')
    parser.add_argument('--atom-one-hot',type=str, default="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/ATOM/atom_one_hot_" ,
                        help='path to one hot atom encoding')
    parser.add_argument('--same-res-atom-neigh',type=str, default="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GRAPHNEIGH/Same_Res_Index_" ,
                        help='path to same residue atom neighbours')
    parser.add_argument('--diff-res-atom-neigh',type=str, default="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GRAPHNEIGH/Diff_Res_Index_" ,
                        help='path to diff residue atom neighbours')
    parser.add_argument('--res-neigh',type=str, default="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GRAPHNEIGH/Residue_Neigh_" ,
                        help='path to residue neighbour')
    parser.add_argument('--path-res-trans',type=str, default="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/TRANS/Trans_" ,
                        help='path to transformer feature')
    parser.add_argument('--res-no',type=str, default="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/Atomfreq/atomfreq_" ,
                        help='path to residue number')
    args = parser.parse_args()
    torch.set_float32_matmul_precision('high')
    Val_label_file=args.val_set
    Train_label_file=args.train_set
    Test_label_file=args.test_set
    gdtts=args.gdtts
    same_res_atom_neigh=args.same_res_atom_neigh
    diff_res_atom_neigh=args.diff_res_atom_neigh
    workers=args.workers
    batch_size=args.batch_size
    res_neigh=args.res_neigh
    path_res_trans=args.path_res_trans
    atom_one_hot=args.atom_one_hot
    res_no=args.res_no
    temp=data1(path_res_trans,same_res_atom_neigh,diff_res_atom_neigh,atom_one_hot,res_neigh,gdtts,batch_size,workers,Train_label_file,Val_label_file,Test_label_file,res_no)
    TD=Test_Dataset(temp)
    test_loader = DataLoader(TD, batch_size=temp.batchSize,shuffle=False,num_workers=int(temp.workers),collate_fn=collate_fn_padd_test)
    # model
    path1=glob.glob(args.model_path)
    print(path1[0])
    model = GNN1().load_from_checkpoint(path1[0])
    model.eval()
    # training
    trainer = pl.Trainer(min_epochs=3,accelerator="gpu", devices=args.devices, max_epochs=args.epochs,num_nodes=args.nodes,enable_checkpointing=False)#,limit_train_batches=10)  #,resume_from_checkpoint=path1)
    #trainer.fit(model, train_loader)
    trainer.test(model,test_loader)

