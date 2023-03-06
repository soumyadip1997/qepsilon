import glob
import pandas as pd
import numpy as np
import os
neigh_atom="Features/GRAPHNEIGH/Same_Res_Index_"
neigh_res="Features/GRAPHNEIGH/Residue_Neigh_"
gdtts="Features/GDT_TS/gdtts_"
trans="Features/TRANS/Trans_"
one_hot_atom="Features/ATOM/atom_one_hot_"
#gdt=glob.glob(gdtts+"*")
df=np.load("NEWCASP13.npy")
casp_ed=np.array(df[:,0])
target_list=np.array(df[:,1])
decoy_list=np.array(df[:,2])
gdt_ts=df[:,3]
res=[]
for i in range(len(casp_ed)):
   #if casp_ed[i]==10: 
       neigh_atom_path=neigh_atom+target_list[i]+"_"+decoy_list[i]+".npy"
       neigh_res_path=neigh_res+target_list[i]+"_"+decoy_list[i]+".npy"
        
       trans_path=trans+target_list[i]+"_"+decoy_list[i]+".npy"
       gdtts_path=gdtts+target_list[i]+"_"+decoy_list[i]+".npy"
       one_hot_atom_path=one_hot_atom+target_list[i]+"_"+decoy_list[i]+".npy"
       try: 
           if os.path.exists(neigh_atom_path)  and os.path.exists(neigh_res_path)  and os.path.exists(trans_path)   and os.path.exists(gdtts_path) and os.path.exists(one_hot_atom_path):
                A=np.load(neigh_atom_path).reshape(2,-1)
                B=np.load(neigh_res_path).reshape(2,-1)
                print(f"Done,{target_list[i]}") 
                res.append([casp_ed[i],target_list[i],decoy_list[i],gdt_ts[i]])
       except:
            print("No")
res=np.array(res).reshape(-1,4)
np.save("Q-epsilon/Test_CASP13_new.npy",res)
    

