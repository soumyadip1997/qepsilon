
import pandas as pd
import os
import warnings
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import glob
from Bio.PDB import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GDTTS')
    parser.add_argument('--decoy-location', type=str, default="Q-epsilon/", metavar='N',
                        help='location to the downloaded decoy 3D structures of all CASP')
    parser.add_argument('--output-location', type=str, default="Q-epsilon/Features/", metavar='O',
                        help='location for the output features to be stored')
    args = parser.parse_args()
    F=open("Failure_atom.txt","a")
    
    output_path=args.output_location+"GDT_TS/"
    CASP_DIR=['CASP9','CASP10','CASP11','CASP12','CASP13','CASP14']
    
    result_name=[]
    for p1 in CASP_DIR:
        loc=glob.glob(args.decoy_location+p1+"/Labels/*")
        for i in loc:
            target_name=(i.split("/")[-1])
            #print(i)
            try:
                A12=pd.read_csv(i,delimiter=r"\s+")
                
                decoy_list=np.array(A12["Model"])
                GDTTS=np.array(A12["GDT_TS"])
                for po in range(len(GDTTS)):
                        gdt_name=output_path+"gdtts_"+target_name.split(".")[0]+"_"+decoy_list[po]
                        np.save(gdt_name,float(GDTTS[po]/100))
                        
                        print(gdt_name,GDTTS[po]/100)
            except:
                print("No")
                    #result_name.append(gdt_name)
                    #dict1={gdt_name:GDTTS[po].tobytes()}
                    #writeCache(env,dict1) 
    '''f = open(i,'r')
                    for lines in f.readlines():
                        if lines.split()[6]!="GDT_TS":
                        decoy=str((lines.split()[0]).split(".")[0])
                            
                        val_dir=val+tar+"/"+decoy
                        #print(val_dir)
                        if os.path.exists(val_dir):



                            score_temp=np.array(float(lines.split()[6])/100).reshape(1,1)
                            req=str(tar+"_"+decoy)
                            print(req)
                            all_files="Names"+str(k11)
                            gdt="GDTTS_"+req
                            #print(gdt,score_temp)      
                            dict1={gdt:score_temp.tobytes(),all_files:req.encode()}
                            writeCache(env,dict1)
                
                            k11+=1
                            #print(req,score_temp)'''
                        
                
        

