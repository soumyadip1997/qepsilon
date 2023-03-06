# QA-with-graph-convolution-guided-by-a-loss-function-designed-for-high-quality-decoys
Downloading the model -  https://colostate-my.sharepoint.com/:u:/r/personal/soumya16_colostate_edu/Documents/best_model.ckpt?csf=1&web=1&e=XqjmLY

Train_Val_Test.zip - It contains all the Training, Validation and Test decoy names along with their targets and CASP version

Feat_script - It contains all the scripts for extracting atom and residue features along with their neighbours.

# Requirements
Python 3.9.16

Numpy

Pandas

Matplotlib

Pytorch 1.13.1

Cuda 11.6

Pytorch Lightning

Pytorch Geometric

# Dataset

The 3D structures of the decoys can be downloaded from - https://zenodo.org/record/7697275#.ZAVRA-zMKrM

# Manual extraction of decoy features

Run on all the decoys 

# Download preprocessed data-

All the features can be downloaded from https://zenodo.org/record/7694318#.ZAVS2-zMKrM and https://colostate-my.sharepoint.com/:u:/g/personal/soumya16_colostate_edu/ESN_lob-izZKm86bIZ39HIQBNWybwmfvJHve-G1394B49Q?e=7oO7E3



# Training 
      python Train.py --batch-size 70 --epochs 50 --workers 12 --seed 42 --devices 1 --nodes 1 --loss-type 0 --train-set Train.npy --val-set Val.npy --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --model-path best_model.ckpt
      
      
Change the --loss-type to 1 and --L1-model to the previous model path (where --loss-type=0)when using the $\epsilon$ modified L1-Loss
      

      
      
# Testing 
      python Test.py --workers 12 --model-path best_model.ckpt --devices 1 --nodes 1 --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --result-file result.csv

# Plotting and calculating Result
      python plot_score.py --result-file result.csv --targets valid_targets_CASP14.csv --plot-name CASP13.pdf


