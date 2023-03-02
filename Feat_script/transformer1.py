import torch
from Bio.PDB.PDBParser import PDBParser
from transformers import BertModel, BertTokenizer,XLNetLMHeadModel, XLNetTokenizer,pipeline,T5EncoderModel, T5Tokenizer
import re
import os
import warnings
import requests
from tqdm.auto import tqdm
import glob
import numpy as np
import os
import pandas as pd
import math
from multiprocessing import Pool
import pickle
def writeCache(env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k.encode(), v)
import lmdb



def transformer1(sequence,model,tokenizer,device):

    ids = tokenizer.batch_encode_plus(sequence, add_special_tokens=True, padding=True)


    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)



    with torch.no_grad():
            embedding = model(input_ids=input_ids,attention_mask=attention_mask)




    embedding = embedding.last_hidden_state.cpu().numpy()



    features = []
    for seq_num in range(len(embedding)):
        seq_len = (attention_mask[seq_num] == 1).sum()
        seq_emd = embedding[seq_num][:seq_len-1]
        features.append(seq_emd)

    return features











def transformer_prep(loc,model,tokenizer,device):
  parser = PDBParser()
  with warnings.catch_warnings(record=True) as w:

    structure = parser.get_structure("1", loc)
  atoms1=[a for a in structure.get_atoms()]
  res_index=np.array([i.get_full_id()[3][1] for i in atoms1])
  unique_res_index=np.array(sorted(np.unique(res_index)))
  for i in range(len(unique_res_index)):
    res_index[np.where(res_index==unique_res_index[i])]=i
  
  residues = [r for r in structure.get_residues()]       
  dist_mat=np.zeros((len(residues),len(residues)))
  taken1=np.zeros((len(residues)))
  
  for i in range(len(residues)):
    one  = residues[i]["CA"].get_coord()

    for j in range(i+1,len(residues)):
      two = residues[j]["CA"].get_coord()
      diff=np.linalg.norm(one-two)
      dist_mat[i][j]=diff
      dist_mat[j][i]=diff

  seq=[]
  res_number=[]
  seq_index=[]
  seq.append(residues[0])
  seq_index.append(0)
  taken1[0]=1
  temp_taken=0
  while(sum(taken1)<len(residues)):
    i=temp_taken
    min1=np.inf
    temp_taken=-1
    flag=0
    for j in range(len(residues)):
      if i!=j:
        if min1>dist_mat[i][j] and taken1[j]==0:
          flag=1
          min1=dist_mat[i][j]
          
          temp_taken=j
    if flag==1:
      taken1[temp_taken]=1
      seq.append(residues[temp_taken])
      seq_index.append(temp_taken)
  res_name1= [f.get_resname() for f in seq]
  temp_name=[""]
  seq_index=np.array(seq_index)
  for k in range(len(res_name1)-1):
    temp_name[0]+=str(res_name1[k])
    temp_name[0]+=" "
  temp_name[0]+=str(res_name1[k])
  trans_feat=transformer1(temp_name,model,tokenizer,device)
  final_trans_feat=np.zeros((len(res_index),len(trans_feat[0][0])))
  for i in range(len(trans_feat[0])):
    final_trans_feat[np.array(np.where(i==res_index)[0])]=trans_feat[0][int(np.where(i==seq_index)[0])]
  return final_trans_feat

def open_lmdb(temp1,env):
    try:
        with env.begin(write=False) as txn:
                temp=txn.get(temp1.encode())
                scores=np.frombuffer(temp)
        return True
    except:
        return False


    
def writeCache(env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k.encode(), v)
if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    f = open("Index.txt", "a")


    model = model.to(device)
    model = model.eval()
    CASP_DIR=["CASP12"]
    data_loc="/s/jawar/f/nobackup/Soumyadip/CASP_DATA/"
    feature_loc="/s/lovelace/c/nobackup/asa/soumya16/QA/CASP_FEAT/Prot/"
    k=0
    output_path=feature_loc+"Decoys/"
    flag=0
    #env = lmdb.open(output_path, map_size=1099511627776*8)
    for p1 in CASP_DIR:
        decoy_loc=glob.glob(data_loc+p1+"/decoys/*/*")
        k=0
        index_names=[]
        for i in decoy_loc:
             try:
                flag=1
                target_name=i.split("/")[9]
                decoy_name=i.split("/")[10]
                req_output_name=str(p1)+"_"+str(target_name)+"_"+str(decoy_name)
                one_hot_res="Transformer_"+req_output_name
                #if open_lmdb(one_hot_res,env)==False: 
                if True:
                    transformer_features=transformer_prep("/s/jawar/f/nobackup/Soumyadip/CASP_DATA/CASP13/decoys/A0953s1-D1/A0953s1TS122_1-D1.pdb",model,tokenizer,device)
                    #dict1={one_hot_res:np.array(transformer_features).tobytes()}
                    #writeCache(env,dict1)
                    #f.write('{} {}\n'.format(req_output_name,len(transformer_features)))
                    print(len(transformer_features))
                    print(f"Done {i}")
                else:
                    print("Already Done {i}")
                k+=1
             except:
                print("No")
             break
    f.close()
