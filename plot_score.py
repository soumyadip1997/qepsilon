import pandas as pd
import argparse

import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(description='PyTorch Lightening QA')
parser.add_argument('--targets',type=str, default="valid_targets_CASP14.csv" ,
                        help='Filename containing all the set of targets')
parser.add_argument('--result-file',type=str, default="result.csv" ,
                        help='Path to the result file')
args=parser.parse_args()
File=args.result_file
df1=pd.read_csv(args.targets,sep=".\t",header=None,engine="python")
targets=np.array(df1[1])
df=np.array(pd.read_csv(File,header=None))
names=np.array(df[:,0])
rs=np.array(df[:,1],dtype=float)
ps=np.array(df[:,2],dtype=float)
print(len(rs))
print(scipy.stats.pearsonr(rs,ps),len(rs))
print(scipy.stats.spearmanr(rs,ps))
seq1=[]
seq2=[]

l11=[]
l22=[]
for i in targets:
    try:
        ls1=[]
        ls2=[]
        for j in range(len(names)):
            if i in names[j]:
                seq1.append(rs[j])
                seq2.append(ps[j])
                ls1.append(rs[j])
                ls2.append(ps[j])
        ls1=np.array(ls1)
        ls2=np.array(ls2)
        #print(len(ls1),len(ls2))        
        l11.append(scipy.stats.pearsonr(ls1,ls2)[0])
        l22.append(scipy.stats.spearmanr(ls1,ls2)[0])
    except:
        print(f"no {i}")
seq1=np.array(seq1)
seq2=np.array(seq2)
print(len(seq1))
print(f'Actual pearson {scipy.stats.pearsonr(seq1,seq2)}')
print(f'Actual spearman {scipy.stats.spearmanr(seq1,seq2)}')
print(f"RMSE:{np.mean((seq1-seq2)**2)**0.5}")
l11=np.array(l11)
print(f'Per target pearson {np.mean(l11)}')
print(f'Per target spearman {np.mean(l22)}')

plt.plot(rs,ps,".",alpha=0.02)
plt.xlabel("Actual Scores")
plt.ylabel("Predicted Scores")
plt.savefig("CASP13_Final.pdf",format="pdf")












