# QA-with-graph-convolution-guided-by-a-loss-function-designed-for-high-quality-decoys


Train_Val_Test.zip - It contains all the Training, Validation and Test decoy names along with their targets and CASP version

Feat_script - It contains all the scripts for extracting atom and residue features along with their neighbours.

# Requirements
Python 3.9.16

Numpy 1.23.5

Pandas 1.5.2

Matplotlib 3.6.2

Pytorch 1.13.1

Cuda 11.6

Pytorch Lightning 1.9.0

Pytorch Geometric 2.2.0

Biopython 1.81

Sklearn 1.2.0

# Dataset

To do manual extraction of the features we need the 3D structures of the decoys. It can be downloaded from the following links-

CASP9 - https://zenodo.org/record/7697275/files/CASP9.zip?download=1

CASP10 - https://zenodo.org/record/7697275/files/CASP10.zip?download=1

CASP11 - https://zenodo.org/record/7697275/files/CASP11.zip?download=1

CASP12 - https://zenodo.org/record/7697275/files/CASP12.zip?download=1

CASP13 - https://zenodo.org/record/7697275/files/CASP13.zip?download=1

CASP14 - https://zenodo.org/record/7697275/files/CASP14.zip?download=1

# Manual extraction of decoy features

To manually extract the features for all the decoys run the scripts in Feat_script directory-

atom_feat.py - Extract one hot encodings for all the atoms of  decoys

gdtscores.py - Extract the gdtts of all  decoys

neigh_atom.py - Extract all the atom neighbours of all  decoys

neigh_res.py - Extract all the residue neighbours of all decoys

res_number.py - Extract the  number of atoms present inside each residue of all decoys

transformer_feat.py - Extract transformer feature for each residue of all  decoys

# Or alternatively download the preprocessed data-

To save time all the above features can be downloaded from the links below-

transformer feature for residue - https://colostate-my.sharepoint.com/:u:/g/personal/soumya16_colostate_edu/ESN_lob-izZKm86bIZ39HIQBNWybwmfvJHve-G1394B49Q?e=7oO7E3

one hot encodings of atoms - https://zenodo.org/record/7694318/files/ATOM.zip?download=1

atom and residue neighbours - https://zenodo.org/record/7694318/files/GRAPHNEIGH.zip?download=1

number of atoms inside each residue - https://zenodo.org/record/7694318/files/Atomfreq.zip?download=1

gdtts - https://zenodo.org/record/7694318/files/GDT_TS.zip?download=1

# Training 

We do a two step training process. 

First we run the GCN with L1-Loss for 50 epochs

      python Train.py --batch-size 70 --epochs 50 --workers 12 --seed 42 --devices 1 --nodes 1 --loss-type 0 --train-set Train.npy --val-set Val.npy --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --save-model best_model_l1.ckpt
      
Then we run it with our $\epsilon$ modified L1-Loss for another 10 epochs using the best model obtained from the L1-Loss

      python Train.py --batch-size 70 --epochs 10 --workers 12 --seed 42 --devices 1 --nodes 1 --loss-type 1 --train-set Train.npy --val-set Val.npy --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --load-model best_model_l1.ckpt --save-model best_model.ckpt
      
# Testing 

To save time we have provided our best model - 

Downloading the model -  https://zenodo.org/record/7697220/files/best_model.ckpt?download=1

After training for a total of 60 epochs or downloading the model run the following-

      python Test.py --workers 12 --model-path best_model.ckpt --devices 1 --nodes 1 --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --result-file result.csv



# Plotting and calculating Result

For calculating the pearson and spearman corelation scores and plotting data run the following -

      python plot_score.py --result-file result.csv --targets valid_targets_CASP14.csv --plot-name CASP13.pdf


