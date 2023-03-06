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
    parser = argparse.ArgumentParser(description='Res_Number_Info')
    parser.add_argument('--decoy-location', type=str, default="Q-epsilon/", metavar='N',
                        help='location to the downloaded decoy 3D structures of all CASP')
    parser.add_argument('--output-location', type=str, default="Q-epsilon/Features/", metavar='O',
                        help='location for the output features to be stored')
    args = parser.parse_args()

    
    output_path=args.output_location+"/ATOMfreq/"
    CASP_DIR=['CASP9','CASP10','CASP11','CASP12','CASP13','CASP14']
    
    for p1 in CASP_DIR:

        req_loc=glob.glob(args.decoy_location+p1+"/decoys/*/*")

        flag=0

        for i in range(len(req_loc)):
            parser = PDBParser()
            with warnings.catch_warnings(record=True) as w:
                structure = parser.get_structure("", req_loc[i])
            l1=res1(structure)
            target_name=req_loc[i].split("/")[-2]
            atomfreq=output_path+"atomfreq_"+target_name+"_"+(req_loc[i].split("/")[-1]).split(".")[0]
                     
            np.save(atomfreq,l1)

        
