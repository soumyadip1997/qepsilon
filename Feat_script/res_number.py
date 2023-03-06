import warnings
import numpy as np
import glob
from Bio.PDB import *
def res1(structure):



    atom_freq=[]
    for residue in structure.get_residues():
        count1=0
        for j in residue.get_atoms():
            count1+=1 
        atom_freq.append(count1)
    atom_freq=np.array(atom_freq)
    return atom_freq
if __name__ == "__main__":
    loc="/s/lovelace/c/nobackup/asa/soumya16/QA_project/"
    CASP_DIR=["CASP13_EXtra"]
    temp_loc="/s/jawar/f/nobackup/Soumyadip/"
    output_path="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/Atomfreq/"

    for p1 in CASP_DIR:

        req_loc=glob.glob(loc+p1+"/decoys/*/*")

        flag=0

        for i in range(len(req_loc)):
            parser = PDBParser()
            with warnings.catch_warnings(record=True) as w:
                structure = parser.get_structure("", req_loc[i])
            l1=res1(structure)
            target_name=req_loc[i].split("/")[-2]
            atomfreq=output_path+"atomfreq_"+target_name+"_"+(req_loc[i].split("/")[-1]).split(".")[0]
                     
            np.save(atomfreq,l1)

        
