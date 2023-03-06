'''
One hot encoded residue infomration using SKlearn Library

Output is N*M where N is the total number of atoms and M is the encoded features of the residues.
Any unknown  residue is mapped to 1
'''
import pandas as pd
from Bio.SeqIO.FastaIO import SimpleFastaParser
from Bio.PDB.NeighborSearch import NeighborSearch
import math
import warnings
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import glob
from Bio.PDB import *
from Bio.PDB.NeighborSearch import NeighborSearch
import lmdb
import math
import numpy as np
from multiprocessing import Pool
def neigh1(i11):
    parser = PDBParser()
    try:
    #if True:
        with warnings.catch_warnings(record=True) as w:
                structure = parser.get_structure("", i11)
        atom_list=np.array([atom for atom in structure.get_atoms()])
        p4=NeighborSearch(atom_list)
        neighbour_list=p4.search_all(6,level="R")
        
        neighbour_list=np.array(neighbour_list,dtype=object)
     
        neigh_residue_list1=np.array([r.id[1] for r in neighbour_list[:,0]])
        neigh_residue_list2=np.array([r.id[1] for r in neighbour_list[:,1]])
        
        residues = np.array([r.id[1] for r in structure.get_residues() if r.get_id()[0] == " "])
        neigh_info=[[-1]*10 for i in range(len(residues)) ]
        for i in range(len(residues)):
            pos1=np.where(residues[i]==neigh_residue_list1)[0]
            pos2=np.where(residues[i]==neigh_residue_list2)[0]
            k=0
            for j in pos1:
                res2=neighbour_list[j][1]
                req_index=np.where(res2.id[1]==residues)[0][0]
                neigh_info[i][k]=req_index
                k+=1
                if k==10:
                    break
            for j in pos2:
                if k==10:
                    break
                res2=neighbour_list[j][0]
                        
                req_index=np.where(res2.id[1]==residues)[0][0]
                if req_index not in neigh_info[i]:
                    neigh_info[i][k]=req_index
                    k+=1
        #print(str(i.split("/")[-2]))
        neigh_index=output_path+"Neigh_numpy_"+str(i11.split("/")[-2])+"_"+str((i11.split("/")[-1]).split(".")[0])+".npy"
        
        np.save(neigh_index,neigh_info)
        
        print(f"Done,{neigh_index}")
        
    except:
        F.write(f"{i11}\n")

if __name__ == "__main__":
    
    output_path="Features/NEIGH_RES/"
    F=open("Failure_neigh_res_CASP9","w+")
    CASP_DIR=['CASP9','CASP10','CASP11','CASP12','CASP13','CASP14']
    for j in CASP_DIR:
        input_structure=glob.glob(j+"/decoys/*/*.pdb")
        k1=0 
        with Pool(30) as p:
            p.map(neigh1,[input_structure[i] for i in range(len(input_structure))])
