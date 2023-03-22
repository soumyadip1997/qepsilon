# qepsilon

[![DOI](https://zenodo.org/badge/607034727.svg)](https://zenodo.org/badge/latestdoi/607034727)


Repository for Protein quality assessment with graph convolution guided by a loss function designed for high quality decoys

# Dependencies

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

torchsampler 0.14.1 

# Initial Setup

First install dependencies in a conda environment and clone the repository-

      git clone https://github.com/soumyadip1997/Q-epsilon.git
           
      cd Q-epsilon


# Data Preprocessing and Train-Val-Test split
It can be done either from 

1. **Scratch** 

or 

2. **Use the already prepared Train,Val and Test set and Preprocessed data**

# From scratch

First we download the 3D structure decoy files.


CASP9 - https://zenodo.org/record/7697275/files/CASP9.zip?download=1

CASP10 - https://zenodo.org/record/7697275/files/CASP10.zip?download=1

CASP11 - https://zenodo.org/record/7697275/files/CASP11.zip?download=1

CASP12 - https://zenodo.org/record/7697275/files/CASP12.zip?download=1

CASP13 - https://zenodo.org/record/7697275/files/CASP13.zip?download=1

CASP14 - https://zenodo.org/record/7697275/files/CASP14.zip?download=1

Then we unzip the folders using 

           unzip <CASP_FILE> -d Q-epsilon/

where <CASP_FILE> represents the downloaded CASP zip files


Then we make a new folder by the name "Features" inside Q-epsilon

           mkdir Features

Make sub directories under Features-

           sh create.sh


Then we extract the features of all the decoys by running the scripts in Feat_script directory-



For extracting one hot encodings of all the atoms of  decoys

      python atom_feat.py --decoy-location Q-epsilon/ --output-location Q-epsilon/Features/

For extracting the gdtts of all  decoys

      python gdtscores.py --decoy-location Q-epsilon/ --output-location Q-epsilon/Features/

For extracting the same and different residue atom neighbours of all  decoys

      python neigh_atom.py --decoy-location Q-epsilon/ --output-location Q-epsilon/Features/

For extracting all the residue neighbours of all decoys

      python neigh_res.py  --decoy-location Q-epsilon/ --output-location Q-epsilon/Features/

For extracting the  number of atoms present inside each residue of all decoys

      python res_number.py  --decoy-location Q-epsilon/ --output-location Q-epsilon/Features/

For extracting transformer feature for each residue of all  decoys
 
      python transformer_feat.py --decoy-location Q-epsilon/ --output-location Q-epsilon/Features/
      
      
For extracting the LDDT of all  decoys run the following script in each of the CASP directory

      python calc_lddt.py --table-loc *.xz --output-location Q-epsilon/Features/ --group-info output.csv
      
**Parameters**

--decoy-location   Location of the unzipped CASP folders
           
--output-location  Location for storing the output features having the required sub-directories

--table-loc        Location of the table.pkl.xz of each CASP

--group-info       Location of group info present in output.csv 

Then we need to create Train,Val and Test Set. Since we are training with both GDTTS and LDDT scores we have two sets of train, validation and test datasets

**GDTTS**

Run the scripts inside Train_Val_Test directory.
      
To create the Train and Validation set first run 

      python create_possible_list.py --same-res-atom-neigh Q-epsilon/Features/GRAPHNEIGH/Same_Res_Index_ --res-neigh Q-epsilon/Features/GRAPHNEIGH/Residue_Neigh_ --gdtts Q-epsilon/Features/GDT_TS/gdtts_ --atom-one-hot Q-epsilon/Features/ATOM/atom_one_hot_ --path-res-trans Q-epsilon/Features/TRANS/Trans_

Then run 

      create_train_val.py --output-path Q-epsilon



To create the Test set first run 

      python process_CASP13_14.py --decoy-location Q-epsilon/
      
Then run

      python create_possible_list_CASP13.py --same-res-atom-neigh Q-epsilon/Features/GRAPHNEIGH/Same_Res_Index_ --res-neigh Q-epsilon/Features/GRAPHNEIGH/Residue_Neigh_ --gdtts Q-epsilon/Features/GDT_TS/gdtts_ --atom-one-hot Q-epsilon/Features/ATOM/atom_one_hot_ --path-res-trans Q-epsilon/Features/TRANS/Trans_ --output-path Q-epsilon
      
      python create_possible_list_CASP14.py --same-res-atom-neigh Q-epsilon/Features/GRAPHNEIGH/Same_Res_Index_ --res-neigh Q-epsilon/Features/GRAPHNEIGH/Residue_Neigh_ --gdtts Q-epsilon/Features/GDT_TS/gdtts_ --atom-one-hot Q-epsilon/Features/ATOM/atom_one_hot_ --path-res-trans Q-epsilon/Features/TRANS/Trans_ --output-path Q-epsilon
 
This will create the test sets for CASP13 and 14 respectively.

**Parameters**

--same-res-atom-neigh - Location of same residue atom neighbours

-res-neigh - Location of residue  neighbours

--gdtts - Location of gdtts

--atom-one-hot - Location of one hot atom embeddings

--path-res-trans - Location of residue features

--output-path - Location for saving the files generated by create_train_val.py 

--decoy-location - Location of the CASP Folders

**LDDT**

Since for LDDT we are using the same target set as DeepUMQA(https://academic.oup.com/bioinformatics/article/38/7/1895/6520805) we are going to use Train_LDDT.npy, Val_LDDT.npy  as the train and validation sets  and Test_CASP13_LDDT.npy and Test_CASP14_LDDT.npy as the test sets.

# Alternative: Downloading the preprocessed data and using the already prepared Train-Val-Test File

To save time all the above features can be downloaded from the links below-

transformer feature for residue - https://colostate-my.sharepoint.com/:u:/g/personal/soumya16_colostate_edu/ESN_lob-izZKm86bIZ39HIQBNWybwmfvJHve-G1394B49Q?e=7oO7E3

one hot encodings of atoms - https://zenodo.org/record/7694318/files/ATOM.zip?download=1

same and diferent residue atom neighbours - https://zenodo.org/record/7694318/files/GRAPHNEIGH.zip?download=1

residue neighbours - https://zenodo.org/record/7703112/files/NEIGH_RES.zip?download=1

number of atoms inside each residue - https://zenodo.org/record/7694318/files/Atomfreq.zip?download=1

gdtts - https://zenodo.org/record/7694318/files/GDT_TS.zip?download=1


LDDT- https://zenodo.org/record/7760929/files/LDDT.zip?download=1

Then unzip the files and store them inside Features Folder

           unzip <Filename.zip> -d Q-epsilon/Features/
           
where <Filename.zip> are the different files that we downloaded (ATOM.zip,ATOMfreq.zip,GDT_TS.zip,GRAPHNEIGH.zip,RES_NEIGH.zip)         

Then use the Train_Val_Test.zip file which contains all the information for the train,validation and test sets.

Unzip the Train_Val_Test.zip. 

      unzip Train_Val_Test.zip -d Q-epsilon/

It contains the following files

Train.npy - Train set

Val.npy - Validation Set

Test_CASP13_new.npy - Test Set with CASP13 decoys

Test_CASP14_new.npy - Test Set with CASP14 decoys

valid_targets_CASP13.npy -  CASP13 targets 

valid_targets_CASP14.npy -  CASP14 targets


# Training


**GDTTS**

Training is done in two steps: 

First the GCN runs with L1-Loss for 50 epochs and the best model is saved as best_model_l1.ckpt

      python Train.py --batch-size 70 --epochs 50 --workers 12 --seed 42 --devices 1 --nodes 1 --loss-type 0 --train-set Train.npy --val-set Val.npy --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --save-model best_model_l1.ckpt
      
Then the same GCN runs with our $\epsilon$ modified L1-Loss for another 10 epochs using the best model obtained from the L1-Loss i.e. best_model_l1.ckpt. The final model is saved as best_model.ckpt

      python Train.py --batch-size 70 --epochs 10 --workers 12 --seed 42 --devices 1 --nodes 1 --loss-type 1 --train-set Train.npy --val-set Val.npy --test-set Test_CASP13_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --load-model best_model_l1.ckpt --save-model best_model.ckpt
      
      
**LDDT**

For training with LDDT scores we use the pretrained GDTTS model.

      python Train_LDDT.py --workers 12 --batch-size 70 --save-model best_model_lddt.ckpt --load-model best_model.ckpt  --devices 1 --lr 0.001 --epochs 50 --loss-type=1 --train-set Train_LDDT.npy --val-set Val_LDDT.npy --gdtts Features/LDDT/LDDT_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_


      
### Parameters
--batch-size  - Batch size 

--epochs - Number of epochs for training 

--workers - Number of workers 

--seed - Setting the seed value of pytorch

--devices - Number of gpus to be used

--nodes - Number of nodes to be used

--loss-type - Specify which loss function to be used i.e. 0 for L1-Loss and 1 for $\epsilon$ modified L1-Loss

--train-set - Train set location

--val set - Validation set location

--test-set - Test set location

--gdtts - Location with the common prefix of GDTTS/LDDT files

--atom-one-hot - Location with the common prefix for the one hot encodings of atom files

--same-res-atom-neigh - Location with the common prefix for the same residue atom neighbour files

--diff-res-atom-neigh - Location with the common prefix for the different residue atom neighbour files

--res-neigh - Location with the common prefix for residue neighbours

--path-res-trans - Location with the common prefix for the transfomer features of residues

--res-no - Location with the common prefix for the Atomfreq features

--load-model - Location to a model for loading

--save-model - Location for saving a model





# Testing 

We use the best model saved by the $\epsilon$ modified L1-Loss network for testing.

To save training time, the best GDTTS model can be downloaded from  https://zenodo.org/record/7697220/files/best_model.ckpt?download=1 and the best LDDT model can be downloaded from https://zenodo.org/record/7760820/files/best_model_LDDT.ckpt?download=1

After training for a total of 60 epochs or downloading the model run the following for testing on CASP13/CASP14-

**GDTTS**

      python Test.py --workers 12 --model-path best_model.ckpt --devices 1 --nodes 1 --test-set Test_CASP13_new.npy/Test_CASP14_new.npy  --gdtts Features/GDT_TS/gdtts_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --result-file result_13_GDTTS.csv/result_14_GDTTS.csv

**LDDT**

      python Test.py --workers 12 --model-path best_model_LDDT.ckpt --devices 1 --nodes 1 --test-set Test_CASP13_LDDT.npy/Test_CASP14_LDDT.npy  --gdtts Features/LDDT/LDDT_ --atom-one-hot Features/ATOM/atom_one_hot_ --same-res-atom-neigh Features/GRAPHNEIGH/Same_Res_Index_  --diff-res-atom-neigh Features/GRAPHNEIGH/Diff_Res_Index_  --res-neigh Features/GRAPHNEIGH/Residue_Neigh_ --path-res-trans Features/TRANS/Trans_ --res-no Features/Atomfreq/atomfreq_ --result-file result_13_LDDT.csv/result_14_LDDT.csv
This saves all the results to the file specified by --result-file argument which in this case is either result_13.csv or result_14.csv. 


### Parameters
--workers - Number of workers

--model-path - Location to a model for loading

--devices - Number of gpus to be used

--nodes - Number of nodes to be used

--test-set - Test set location

--gdtts - Location with the common prefix of GDTTS/LDDT files


--atom-one-hot - Location with the common prefix for the one hot encodings of atom files

--same-res-atom-neigh - Location with the common prefix for the same residue atom neighbour files

--diff-res-atom-neigh - Location with the common prefix for the different residue atom neighbour files

--res-neigh - Location with the common prefix for residue neighbours

--path-res-trans - Location with the common prefix for the transfomer features of residues

--res-no - Location with the common prefix for the Atomfreq features


--result-file - Location of the file with filename for storing result

# Plotting and calculating Result

For calculating the pearson and spearman correlation scores and plotting data run the following -

**GDTTS**

      python plot_score.py --result-file result_13_GDTTS.csv/result_14_GDTTS.csv --targets valid_targets_CASP14.csv/valid_targets_CASP13.csv --plot-name CASP13.pdf/CASP14.pdf
      
**LDDT**      
      
      python plot_score.py --result-file result_13_LDDT.csv/result_14_LDDT.csv --targets targets_V13.csv/targets_V14.csv --plot-name CASP13.pdf/CASP14.pdf
      
### Parameters

--result-file - Location to the file produced by Test.py


--targets - Location to the file that has target information

--plot-name - Location to save the plots
 


