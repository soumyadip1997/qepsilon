import pandas as pd
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
File="Result_CASP13_new.csv"
#File="Result_old.csv"
df1=pd.read_csv("valid_targets1.csv",sep=".\t",header=None,engine="python")
targets=np.array(df1[1])
df=np.array(pd.read_csv(File,header=None))
names=np.array(df[:,0])
print(len(targets))
rs=np.array(df[:,1],dtype=float)
ps=np.array(df[:,2],dtype=float)
print(rs)
print(ps)
print(scipy.stats.pearsonr(rs,ps))
print(scipy.stats.spearmanr(rs,ps))
seq1=[]
seq2=[]
names1=[]
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
                names1.append(names[j])
        ls1=np.array(ls1)
        ls2=np.array(ls2)
        #print(len(ls1),len(ls2))        
        l11.append(scipy.stats.pearsonr(ls1,ls2)[0])
        l22.append(scipy.stats.spearmanr(ls1,ls2)[0])
    except:
        print(f"no {i}")
seq1=np.array(seq1)
seq2=np.array(seq2)
names1=np.array(names1)
print(f'Actual pearson {scipy.stats.pearsonr(seq1,seq2)}')
print(f'Actual spearman {scipy.stats.spearmanr(seq1,seq2)}')
print(f"RMSE:{np.mean((seq1-seq2)**2)**0.5}")
l11=np.array(l11)
print(f'Per target pearson {np.mean(l11)}')
print(f'Per target spearman {np.mean(l22)}')
num_samples=4000
idx = np.random.choice(np.arange(len(seq1)), num_samples)
'''
plt.plot(seq1[idx],seq2[idx],".")
plt.xlabel("True GDTTS")
plt.ylabel("Predicted GDTTS")
plt.savefig("CASP13.pdf",format="pdf")'''
top_rs=seq1[seq1>=0.90]
top_ps=seq2[seq1>=0.90]
top_names_new=names1[seq1>=0.90]
diff1=np.mean(np.abs((top_rs-top_ps)))
print(diff1,len(top_rs))
path2="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/Atomfreq/atomfreq_"
len1=[]
len_list=[]
diff_list=[[],[],[],[],[]]#0-100,100-250,250-
for i in range(len(top_names_new)):
    key=top_names_new[i]
    diff12=(abs(top_rs[i]-top_ps[i]))
    len1=len(np.load(path2+key+".npy"))
    len_list.append(len1)
    if len1>=0 and len1<=100:
        diff_list[0].append(diff12)
    #elif len1>50 and len1<=100:
    #    diff_list[1].append(diff12)
    elif len1>100 and len1<=150:
        diff_list[1].append(diff12)
    elif len1>150 and len1<=200:
        diff_list[2].append(diff12)
    elif len1>200 and len1<=250:
        diff_list[3].append(diff12)
    #elif len1>250 and len1<=300:
    #    diff_list[5].append(diff12)
    else:
        diff_list[4].append(diff12)
    #print(top_rs[i],top_ps[i],len(np.load(path1+key+".npy")),len(np.load(path2+key+".npy")))
final_diff=np.array([0]*len(diff_list),dtype=np.float64)
for i in range(len(diff_list)):
    final_diff[i]=np.mean(diff_list[i])
'''plt.plot(top_rs,top_ps,".")
plt.xlabel("Actual Scores  ")
plt.ylabel("Predicted Scores")
plt.savefig("CASP13_9.pdf",format="pdf")'''
print(np.unique(len_list))
my_xticks = ['R<=100','100<R<=150','150<R<=200','200<R<=250','R>250']
x=np.array([0,1,2,3,4])
print(final_diff)
plt.xticks(x,my_xticks,fontsize=8)
plt.plot(x,final_diff,".")
plt.ylabel("Difference between True and Predicted GDTTS among decoys")
plt.savefig("CASP13_9_diff.pdf",format="pdf")
