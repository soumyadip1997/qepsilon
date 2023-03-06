import numpy as np
A=np.load("Train_Val.npy")
train_set=["9","10","11"]
B=np.empty((1,4))
for j in train_set:
    pos1=np.where(A[:,0]==j)[0]
    B=np.concatenate((B,A[pos1]),axis=0)
B=B[1:]      
pos1=np.where(A[:,0]=="12")[0]
len1=int(len(pos1)*0.80)
casp12=A[pos1]
B=np.concatenate((B,casp12[:len1]),axis=0)
np.save("Train.npy",B)
val=casp12[len1:]
np.save("Val.npy",val)

