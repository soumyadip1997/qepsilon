import glob
import pandas as pd
import numpy as np
import os
neigh_atom="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GRAPHNEIGH/Same_Res_Index_"
neigh_res="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GRAPHNEIGH/Residue_Neigh_"
gdtts="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/GDT_TS/gdtts_"
trans="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/TRANS/Trans_"
one_hot_atom="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/ATOM/atom_one_hot_"
#gdt=glob.glob(gdtts+"*")
df=pd.read_csv("/s/lovelace/c/nobackup/asa/soumya16/QA_project/decoys.csv")
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
    

