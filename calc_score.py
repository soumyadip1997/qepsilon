import pandas as pd
import argparse

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch Lightening QA')
parser.add_argument('--decoys',type=str, default="CASP14_LDDT.csv" ,
                        help='Filename containing all the set of decoys')
parser.add_argument('--result-file',type=str, default="Result_CASP14_LDDT.csv" ,
                        help='Path to the result file')
parser.add_argument('--plot-name',type=str, default="CASP13_Final.pdf" ,
                        help='Name of the plot in pdf format')
args=parser.parse_args()
File1=args.result_file
df1=pd.read_csv(args.decoys,header=None,engine="python")
decoys=np.unique(np.array(df1[0]))
print(len(decoys),File1)
df=np.array(pd.read_csv(File1,header=None,engine="python"))
names=np.array(df[:,0])
rs=np.array(df[:,1],dtype=float)
ps=np.array(df[:,2],dtype=float)
#print(len(rs))
seq1=[]
seq2=[]
final_names=[]
l11=[]
l22=[]
count=0
for i in decoys:
    try:
        ls1=[]
        ls2=[]
        for j in range(len(names)):
            if i in names[j] and names[j] not in final_names:
                seq1.append(rs[j])
                seq2.append(ps[j])
                final_names.append(names[j])
                break
    except:
        print(f"no {i}")
seq1=np.array(seq1)
seq2=np.array(seq2)
print(len(seq1))
print(f'Actual pearson {scipy.stats.pearsonr(seq1,seq2)}')
print(f'Actual spearman {scipy.stats.spearmanr(seq1,seq2)}')
print(f"RMSE:{np.mean((seq1-seq2)**2)**0.5}")
