'''
Using Sklearn One hot encoder to encode the atoms
Output is of size N*M where N is the total number of atoms and M is the total number of encoded features

'''
import warnings
from Bio.PDB import *
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import glob
import os


def atom1(structure):
    atomslist=np.array(sorted(np.array(['0','1','2','3','4','5','6','7','8','9','10','11']))).reshape(-1,1)
    enc = OneHotEncoder(handle_unknown='ignore')
    dict_atom={'CYS_SG':'0','MET_SD':'0','HIS_CD2':'1','HIS_CE1':'1','HIS_CG':'1','PHE_CD1':'1','PHE_CD2':'1','PHE_CE1':'1','PHE_CE2':'1','PHE_CG':'1','PHE_CZ':'1','TRP_CD1':'1','TRP_CD2':'1','TRP_CE2':'1','TRP_CE3':'1','TRP_CG':'1','TRP_CH2':'1','TRP_CZ2':'1','TRP_CZ3':'1','TYR_CD1':'1','TYR_CD1':'1','TYR_CD2':'1','TYR_CE1':'1','TYR_CE2':'1','TYR_CG':'1','TYR_CZ':'1','ARG_CZ':'2','ASN_CG':'2','ASP_CG':'2','GLN_CD':'2','GLU_CD':'2','C':'2','CA':'3','N':'4','ASN_ND2':'4','GLN_NE2':'4','ARG_NH1':'5','ARG_NH2':'5','ARG_NE':'5','LYS_NZ':'6','SER:OG':'7','THR_OG1':'7','TYR_OH':'7','ASN_OD1':'8','GLN_OE1':'8','ASP_OD1':'9','ASP_OD2':'9','GLU_OE1':'9','GLU_OE2':'9' ,'ALA_CB':'10','ARG_CB':'10','ARG_CG':'10','ASN_CB':'10','ASP_CB':'10','GLN_CB':'10','GLN_CG':'10','GLU_CB':'10','GLU_CG':'10','HIS_CB':'10','ILE_CB':'10','ILE_CD1':'10','ILE_CG1':'10','ILE_CG2':'10','LEU_CB':'10','LEU_CD1':'10','LEU_CD2':'10','LEU_CG':'10','LYS_CB':'10','LYS_CD':'10','LYS_CG':'10','MET_CB':'10','PHE_CB':'10','PRO_CB':'10','PRO_CG':'10','SER_CB':'10','THR_CG2':'10','TRP_CB':'10','TYR_CB':'10','VAL_CB':'10','VAL_CG1':'10','VAL_CG2':'10'}


    keys_list=dict_atom.keys()
    enc.fit(atomslist)
    atom_list=[]
    for atom in structure.get_atoms():
    
        atom_name=atom.get_name()
        res_name=atom.get_parent().get_resname()
        temp_name=res_name+"_"+atom_name
            
        if atom_name=='CA' or atom_name=='N' or atom_name=='C':
            atom_list.append(dict_atom[atom_name])
        elif temp_name in keys_list:
            
            atom_list.append(dict_atom[temp_name])
        else:
            atom_list.append('11')
    atoms_onehot=enc.transform(np.array(atom_list).reshape(-1,1)).toarray()
    return atoms_onehot
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Atom_Feat')
    parser.add_argument('--decoy-location', type=str, default="Q-epsilon/", metavar='N',
                        help='location to the downloaded decoy 3D structures of all CASP')
    parser.add_argument('--output-location', type=str, default="Q-epsilon/Features/", metavar='N',
                        help='location for the output features to be stored')
    args = parser.parse_args()
    F=open("Failure_atom.txt","a")
    
    output_path=args.output_location+"ATOM/atom_one_hot_"
    CASP_DIR=['CASP9','CASP10','CASP11','CASP12','CASP13','CASP14']
    for p1 in CASP_DIR:
        loc=glob.glob(args.decoy_location+p1+"/decoys/*/*")
    
 
 
        for i in loc:
             tar=i.split("/")[-2]
             decoy=(i.split("/")[-1]).split(".")[0]
             print(tar,decoy)
             try:
             #if True:
                 one_hot_atom=str(output_path+tar+"_"+decoy)
                 if os.path.exists(one_hot_atom):
                    print("exist")
                 else:
                     parser = PDBParser()
                     with warnings.catch_warnings(record=True) as w:
                         structure = parser.get_structure("", i)
     
                     t1=atom1(structure)
                     #print(t1[:10])      
                     np.save(one_hot_atom,np.array(t1)) 
                     print(f"Done {one_hot_atom}")
                
             except:
                 print("no")
             #    F.write(i+'\n')
