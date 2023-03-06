import glob
import pandas as pd
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='Test_13')
parser.add_argument('--same-res-atom-neigh',type=str, default="Q-epsilon/Features/GRAPHNEIGH/Same_Res_Index_" ,
                        help='path to same residue atom neighbours')
parser.add_argument('--res-neigh',type=str, default="Q-epsilon/Features/GRAPHNEIGH/Residue_Neigh_" ,
                        help='path to residue neighbour')
parser.add_argument('--gdtts',type=str, default="Q-epsilon/Features/GDT_TS/gdtts_" ,
                        help='path to gdtts')
parser.add_argument('--atom-one-hot',type=str, default="Q-epsilon/Features/ATOM/atom_one_hot_" ,
                        help='path to one hot atom encoding')
parser.add_argument('--path-res-trans',type=str, default="Q-epsilon/Features/TRANS/Trans_" ,
                        help='path to transformer feature')
parser.add_argument('--output-path',type=str, default="Q-epsilon/" ,
                        help='Output path')
args = parser.parse_args()
neigh_atom=args.same_res_atom_neigh
neigh_res=args.res_neigh
gdtts=args.gdtts
trans=args.path_res_trans
one_hot_atom=args.atom_one_hot

#gdt=glob.glob(gdtts+"*")
df=np.load("NEWCASP14.npy")
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
       #tm_path=tm+target_list[i]+"_"+decoy_list[i]+".npy"
       #cad_path=cad+target_list[i]+"_"+decoy_list[i]+".npy"
       gdtts_path=gdtts+target_list[i]+"_"+decoy_list[i]+".npy"
       #gdtha_path=gdtha+target_list[i]+"_"+decoy_list[i]+".npy"
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
np.save(args.output_path+"Test_CASP14_new.npy",res)
    

