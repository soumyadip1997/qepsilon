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

from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.Polypeptide import *


def transformer1(sequence,model=None,tokenizer=None,device=None):

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











def transformer_prep(loc,model=None,tokenizer=None,device=None):
    #try:
      parser = PDBParser()
      with warnings.catch_warnings(record=True) as w:

        structure = parser.get_structure("1", loc)
      
      residues = [r.get_resname() for r in structure.get_residues()]
      req_seq=""
      for p1 in range(len(residues)):
        req_seq+=str(three_to_one(residues[p1])+" ")
      trans_feat=transformer1([req_seq[:-1]],model,tokenizer,device)
      return np.array(trans_feat)
    #except:
    #    print("No")  
    

    
if __name__ == "__main__":
    tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
    model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")

    device =torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    f = open("Index.txt", "a")

    model = model.to(device)
    model = model.eval()
    #model=None
    #tokenizer=None
    CASP_DIR=["CASP10"]
    output_loc="/s/lovelace/c/nobackup/asa/soumya16/QA_project/Features/TRANS/"
    data_loc="/s/lovelace/c/nobackup/asa/soumya16/QA_project/"
    for p1 in CASP_DIR:
        decoy_loc=glob.glob(data_loc+p1+"/decoys/*/*")
        for i in decoy_loc:
             try:
                flag=1
                target_name=(i.split("/")[-2])
                decoy_name=(i.split("/")[-1]).split(".")[0]
                req_output_name=output_loc+"Trans_"+str(target_name)+"_"+str(decoy_name)
                transformer_features=transformer_prep(i,model,tokenizer,device)
                #print(transformer_features)
                np.save(req_output_name,transformer_features)
                print(f"Done {req_output_name}")
                #break 
             
             except:
                print("No")
        #break
    f.close()
