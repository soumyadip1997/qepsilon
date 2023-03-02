# QA-with-graph-convolution-guided-by-a-loss-function-designed-for-high-quality-decoys
Downloading the model -  https://colostate-my.sharepoint.com/:u:/r/personal/soumya16_colostate_edu/Documents/best_model.ckpt?csf=1&web=1&e=XqjmLY

Train_Val_Test.zip - It contains all the Training, Validation and Test decoy names along with their targets and CASP version

Feat_script - It contains all the scripts for extracting atom and residue features along with their neighbours.

importdatageo.py - It contains the  training,validation and testing dataloader



# Training 
      python Train.py 
# Testing 

python Test.py 

# Plotting the scores and calculating correlation

python plot_score.py 


