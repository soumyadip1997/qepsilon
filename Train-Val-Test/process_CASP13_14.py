'''
One hot encoded residue infomration using SKlearn Library

Output is N*M where N is the total number of atoms and M is the encoded features of the residues.
Any unknown  residue is mapped to 1
'''
import pandas as pd
import os
import warnings
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import glob
from Bio.PDB import *

def writeCache(env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k.encode(), v)

if __name__ == "__main__":
    CASP_DIR=["CASP13","CASP14"]

    for p1 in CASP_DIR:
        loc=glob.glob(p1+"/Labels/*")
        for i in loc:
            result=[]
            target_name=(i.split("/")[-1])
            #print(i)
            try:
                A12=pd.read_csv(i,delimiter=r"\s+")
                
                decoy_list=np.array(A12["Model"])
                GDTTS=np.array(A12["GDT_TS"])
                for po in range(len(GDTTS)):
                        result.append(["14",target_name.split(".")[0],decoy_list[po],float(GDTTS[po]/100)])
            except:
                print("No")
        result=np.array(result).reshape(-1,4)
        np.save("NEW"+p1+".npy",result)
