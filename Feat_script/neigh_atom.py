
'''
It calculates the neighbours of each atom i.e. 10 distinct neighbours
Output is  in the form of a ditionary representing an  adjacency list where each source atom and neighbouring atom is represented bby its sequence index .
'''
import os
import warnings
from Bio.PDB import *
import numpy as np
from Bio.PDB.NeighborSearch import NeighborSearch
import math
import argparse

def neigh1(structure,i):
  try:
    #atom_list is a numpy array  that   contains all the atoms of the pdb file in atom object
    atom_list=np.array([atom for atom in structure.get_atoms()])

    residues = np.array([r.id[1] for r in structure.get_residues() if r.get_id()[0] == " "]) 
    p4=NeighborSearch(atom_list)
    neighbour_list_atom=p4.search_all(6,level="A")
    neighbour_list_residue=p4.search_all(6,level="R")
    #neighbour_list_atom=np.array(neighbour_list_atom)
    #neighbour_list_residue=np.array(neighbour_list_residue)
    atom_number=[]
    residue_number=[]
    for i in atom_list:
        atom_number.append(i.get_serial_number())
    atom_number=np.array(atom_number)
    residues=np.array(residues)
    same_res_neigh_list=[[],[]]
    diff_res_neigh_list=[[],[]]
    res_neigh=[[],[]]
    for i in neighbour_list_atom:
        #print(i[0],i[1])
        if i[0].get_parent().get_id()[1]==i[1].get_parent().get_id()[1]:
            same_res_neigh_list[0].append(np.where(atom_number==i[0].get_serial_number())[0][0])
            same_res_neigh_list[0].append(np.where(atom_number==i[1].get_serial_number())[0][0])
            same_res_neigh_list[1].append(np.where(atom_number==i[1].get_serial_number())[0][0])
            same_res_neigh_list[1].append(np.where(atom_number==i[0].get_serial_number())[0][0])
        else:
            diff_res_neigh_list[0].append(np.where(atom_number==i[0].get_serial_number())[0][0])
            diff_res_neigh_list[0].append(np.where(atom_number==i[1].get_serial_number())[0][0])
            diff_res_neigh_list[1].append(np.where(atom_number==i[1].get_serial_number())[0][0])
            diff_res_neigh_list[1].append(np.where(atom_number==i[0].get_serial_number())[0][0])
    same_res_neigh_list=np.array(same_res_neigh_list).reshape(2,-1)    
    diff_res_neigh_list=np.array(diff_res_neigh_list).reshape(2,-1)    
    
    for i in neighbour_list_residue:
        res_neigh[0].append(np.where(residues==i[0].id[1])[0][0])
        res_neigh[0].append(np.where(residues==i[1].id[1])[0][0])
        res_neigh[1].append(np.where(residues==i[1].id[1])[0][0])
        res_neigh[1].append(np.where(residues==i[0].id[1])[0][0])
     
    res_neigh=np.array(res_neigh).reshape(2,-1)    
    return same_res_neigh_list,diff_res_neigh_list,res_neigh,0
  except:
                    F.write(i+'\n')
                    return -1,-1,-1,1
 
def help1(i,output_path):
                    print(i)
                #try:
                    #print(i) 
                    flag=1
                    parent_dir1=i.split("/")[7]
                    tar=i.split("/")[8]
                    decoy=i.split("/")[9]
                    parser = PDBParser()
                    with warnings.catch_warnings(record=True) as w:
                        structure = parser.get_structure("", i)
                    target_name=i.split("/")[-2]
                    neigh_index_same_res=output_path+"Same_Res_Index_"+target_name+"_"+(i.split("/")[-1]).split(".")[0]
                    neigh_index_diff_res=output_path+"Diff_Res_Index_"+target_name+"_"+(i.split("/")[-1]).split(".")[0]
                    Res_neigh=output_path+"Residue_Neigh_"+target_name+"_"+(i.split("/")[-1]).split(".")[0]
                    if os.path.exists(neigh_index_same_res+".npy"):
                        print(f'exist {i}')
                    else:
                        print(i)
                        t1,t2,t3,t4=(neigh1(structure,i))
                        if t4==0:
                            np.save(neigh_index_same_res,t1)
                            np.save(neigh_index_diff_res,t2)
                            np.save(Res_neigh,t3)
                            print(f"Done, {neigh_index_same_res}")
                            
                #except:

                #    print("no")#pass
       








import glob
#from multiprocessing import Pool

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Neigh_Info')
    parser.add_argument('--decoy-location', type=str, default="Q-epsilon/", metavar='N',
                        help='location to the downloaded decoy 3D structures of all CASP')
    parser.add_argument('--output-location', type=str, default="Q-epsilon/Features/", metavar='O',
                        help='location for the output features to be stored')
    args = parser.parse_args()

    
    output_path=args.output_location+"/GRAPHNEIGH/"
    F=open("FAILURE_NEIGH","w+")

    CASP_DIR=['CASP9','CASP10','CASP11','CASP12','CASP13','CASP14']
    for p1 in CASP_DIR:

        req_loc=glob.glob(args.decoy_location+p1+"/decoys/*/*")
        
        flag=0
        
        for i in range(len(req_loc)):
          help1(req_loc[i],output_path)
        
       
        #with Pool(30) as p:
        #    p.map(help1,[req_loc[i] for i in range(len(req_loc))])
        #for i in range(len(loc)):
        #    help1(loc[i],env)
