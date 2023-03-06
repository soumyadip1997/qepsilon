# QA-with-graph-convolution-guided-by-a-loss-function-designed-for-high-quality-decoys
Downloading the model -  https://zenodo.org/record/7697220/files/best_model.ckpt?download=1


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

To save time you can download all the above features 

transformer feature for residue - https://colostate-my.sharepoint.com/:u:/g/personal/soumya16_colostate_edu/ESN_lob-izZKm86bIZ39HIQBNWybwmfvJHve-G1394B49Q?e=7oO7E3

one hot encodings of atoms - https://zenodo.org/record/7694318/files/ATOM.zip?download=1

atom and residue neighbours - https://zenodo.org/record/7694318/files/GRAPHNEIGH.zip?download=1

number of atoms inside each residue - https://zenodo.org/record/7694318/files/Atomfreq.zip?download=1

gdtts - https://zenodo.org/record/7694318/files/GDT_TS.zip?download=1

# Training 
      python Train.py --batch-size 70 --epochs 50 --workers 12 --seed 42 --devices 1 --nodes 1 --loss-type 0 --train-set Train.npy --val-set Val.npy --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --model-path best_model.ckpt
      
      
Change the --loss-type to 1 and --L1-model to the previous model path (where --loss-type=0)when using the $\epsilon$ modified L1-Loss
      

      
      
# Testing 
      python Test.py --workers 12 --model-path best_model.ckpt --devices 1 --nodes 1 --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --result-file result.csv

# Plotting and calculating Result
      python plot_score.py --result-file result.csv --targets valid_targets_CASP14.csv --plot-name CASP13.pdf


