# QA-with-graph-convolution-guided-by-a-loss-function-designed-for-high-quality-decoys
Downloading the model -  https://colostate-my.sharepoint.com/:u:/r/personal/soumya16_colostate_edu/Documents/best_model.ckpt?csf=1&web=1&e=XqjmLY

Train_Val_Test.zip - It contains all the Training, Validation and Test decoy names along with their targets and CASP version

Feat_script - It contains all the scripts for extracting atom and residue features along with their neighbours.

importdatageo.py - It contains the  training,validation and testing dataloader



# Training 
      python Train.py --batch-size 70 --epochs 50 --workers 12 --seed 42 --devices 1 --nodes 1 --loss-type 0 --train-set Train.npy --val-set Val.npy --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --model-path best_model.ckpt
      
      
      
      
# Testing 
      python Test.py --workers 12 --model-path best_model.ckpt --devices 1 --nodes 1 --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --result_file result.csv

# Plotting and calculating Result
      python plot_score.py --result_file result.csv --targets valid_targets_CASP14.csv


