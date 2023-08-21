import glob
from Bio import  PDB
import os
import numpy as np
from Bio.PDB import PDBIO, Select
from Bio.PDB.PDBParser import PDBParser
class ResidueRangeSelect(Select):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def accept_residue(self, residue):
        return self.start <= residue.id[1] <= self.end
target_dir="targets/*"
curr_decoy_dir="decoys/"
new_decoy_dir="decoys_modified/"
all_targets=sorted(glob.glob(target_dir))
io = PDBIO()
target_list_processed=[]
for target in all_targets:
    target_folder=target.split(".")[0].split("-")[0]+"-D"
    target_loc=glob.glob(target.split(".")[0].split("-")[0]+"*")
    if len(target_loc)==2 and target_loc[0]=="targets/T1137s1-D2.pdb":
        target_loc=target_loc[::-1]
    if len(target_loc)==3:
        target_loc=target_loc[::-1]
    line_list=[]
    for unique_target in target_loc:
        if unique_target not in target_list_processed:
            target_list_processed.append(unique_target) 
            target_name=unique_target.split("/")[-1].split(".")[0]
            line1=0
            p = PDB.PDBParser(QUIET=True)
            structure = p.get_structure("dsa", unique_target)
            residues = np.array([r.id[1] for r in structure.get_residues() if r.get_id()[0] == " "])
            line_list.append(len(residues))
            loc=new_decoy_dir+target_name+"/"
            cmd1="mkdir "+loc
            if os.path.exists(loc)==False:
                os.system(cmd1)


    #PFRMAT TS
    #TARGET T1104
    #MODEL 2
    #PARENT N/A
    #ATOM      1  N   GLN     1      -6.359  18.866  -9.712  1.00 78.86           N


    if len(line_list)>0:
        decoy_file_list=glob.glob(curr_decoy_dir+target_name.split("-")[0]+"/*")
        count=1
        for decoy_file in decoy_file_list:
            print(decoy_file)
            p = PDB.PDBParser(QUIET=True)
            
            structure = p.get_structure("dsa", decoy_file)
            #residues = [r for r in structure.get_residues() if r.get_id()[0] == " "]
            low=0
            
            for line_range in range(1,len(line_list)+1):
                #curr_len=0
                #pre_info=f"PFRMAT TS\nTARGET "+str(target_folder.split("/")[1]+str(line_range))+"\nPARENT N/A\n"
                new_decoy_file=new_decoy_dir+target_folder.split("/")[1]+str(line_range)+"/"+decoy_file.split("/")[-1]+"-D"+str(line_range)+".pdb"
                high=low+line_list[line_range-1]
                io.set_structure(structure)
                io.save(new_decoy_file, ResidueRangeSelect(low,high))
                print(f'--->{high},{len(residues)}')
                '''with open(new_decoy_file,"w") as outfile:
                    outfile.write(pre_info)
                    
                    for r_index in range(low,high):
                        current_residue=residues[r_index]
                         
                        for atom in current_residue:
                            atom_name=atom.get_name()
                            atom_number=atom.get_serial_number()
                            atom_residue=atom.get_parent().get_resname()  
                            atom_coord=atom.get_coord()
                            atom_bfactor=atom.get_bfactor()
                            atom_res_id=atom.get_full_id()[3][1]
                            atom_id=atom.get_id()
                            atom_occupancy=atom.get_occupancy()
                            atom_info=["ATOM",str(atom_number),atom_name,atom_residue,str(atom_res_id),str(atom_coord[0]),str(atom_coord[1]),str(atom_coord[2]),str(atom_occupancy),str(atom_bfactor),str(atom_id)]
                            str1=""
                            str1=atom_info[0]+"      "+atom_info[1]+"  "+atom_info[2]+"   "+atom_info[3]+"     "+atom_info[4]+"      "+atom_info[5]+"  "+str(atom_info[6])+"  "+str(atom_info[7])+"  "+str(atom_info[8])+" "+str(atom_info[9])+"           "+atom_info[10]
                            outfile.write(str1)
                            outfile.write("\n") 

                        
                    outfile.write("TER\n") 
                    outfile.write("END\n")            
                outfile.close()'''
                low=high


                
    '''target_name=target.split("/")[-1].split(".")[0]
    flag=0
    line1=0
    with open(target,"r") as outfile:
            data = outfile.readlines()
            for line in data:
                if "ATOM" in line:
                    line1+=1
    loc=new_decoy_dir+target_name+"/"
        
    cmd1="mkdir "+loc
    if os.path.exists(loc)==False:
        os.system(cmd1) 
    decoy_file_list=glob.glob(curr_decoy_dir+target_name.split("-")[0]+"/*")
    for old_decoy_file in decoy_file_list:
        print(old_decoy_file)'''
    '''for old_decoy in old_decoy_files:
        old_decoy_name=old_decoy.split("/")[-1]
        new_decoy_loc1=loc1+old_decoy_name+"-D1"
        new_decoy_loc2=loc2+old_decoy_name+"-D2"
    '''    
