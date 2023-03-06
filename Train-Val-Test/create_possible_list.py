import glob
import pandas as pd
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='List_train_val')
parser.add_argument('--same-res-atom-neigh',type=str, default="Q-epsilon/Features/GRAPHNEIGH/Same_Res_Index_" ,
                        help='path to same residue atom neighbours')
parser.add_argument('--res-neigh',type=str, default="Q-epsilon/Features/GRAPHNEIGH/Residue_Neigh_" ,
                        help='path to residue neighbour')
parser.add_argument('--gdtts',type=str, default="Features/GDT_TS/gdtts_" ,
                        help='path to gdtts')
parser.add_argument('--atom-one-hot',type=str, default="Features/ATOM/atom_one_hot_" ,
                        help='path to one hot atom encoding')
parser.add_argument('--path-res-trans',type=str, default="Features/TRANS/Trans_" ,
                        help='path to transformer feature')
args = parser.parse_args()
 
neigh_atom=args.same_res_atom_neigh
neigh_res=args.res_neigh
gdtts=args.gdtts
trans=args.path_res_trans
one_hot_atom=args.atom_one_hot
#gdt=glob.glob(gdtts+"*")
df=pd.read_csv("decoys.csv")
casp_ed=np.array(df["casp_ed"])
target_list=np.array(df["target_id"])
decoy_list=np.array(df["decoy_id"])
res=[]
for i in range(len(casp_ed)):
   #if casp_ed[i]==12: 
       neigh_atom_path=neigh_atom+target_list[i]+"_"+decoy_list[i]+".npy"
       neigh_res_path=neigh_res+target_list[i]+"_"+decoy_list[i]+".npy"
       trans_path=trans+target_list[i]+"_"+decoy_list[i]+".npy"
       gdtts_path=gdtts+target_list[i]+"_"+decoy_list[i]+".npy"
       one_hot_atom_path=one_hot_atom+target_list[i]+"_"+decoy_list[i]+".npy"
       if  os.path.exists(one_hot_atom_path)  and  os.path.exists(trans_path) and  os.path.exists(neigh_res_path) and os.path.exists(neigh_atom_path) and os.path.exists(gdtts_path) and os.path.exists(cad_path) and os.path.exists(gdtha_path) and os.path.exists(tm_path):
            print(f"Done,{target_list[i]}")
            score=np.load(gdtts_path) 
            res.append([casp_ed[i],target_list[i],decoy_list[i],score])
res=np.array(res).reshape(-1,4)
np.save("Train_Val.npy",res)
    

