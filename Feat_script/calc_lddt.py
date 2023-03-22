import numpy as np
import pickle 
import glob
parser = argparse.ArgumentParser(description='LDDT')
parser.add_argument('--table-loc', type=str, default="*.xz", metavar='N',
                    help='location to table.xz location of each CASP directory')
parser.add_argument('--output-location', type=str, default="Q-epsilon/Features/", metavar='O',
                    help='location for the output features to be stored')
parser.add_argument('--group-info', type=str, default="output.csv", metavar='O',
                    help='location for the group to number mapping of each CASP')
args = parser.parse_args()

    



file1=glob.glob(args.table_loc)[0]
print(file1)
import lzma
import pandas as pd
map_file=pd.read_csv(args.group_info)#,error_bad_lines=False)
with lzma.open(file1, 'rb') as f:
    obj = pickle.load(f)
Target=(obj["#"])
target_decoy=Target.index
target_values=np.array(obj['LDDT'])
map_id=np.array(map_file["qa_group_id"])
map_name=np.array(map_file["qa_group_name"])
print(map_file)
len1=len(target_values)
path1=args.output_location
for i in range(len1):
    all1=target_decoy[i]
    target_name=str(all1[0])
    print(all1)
    decoy_name=str(map_name[np.where(all1[1]==map_id)[0][0]])
    no=str(all1[2])
    label_name=path1+"LDDT_"+target_name+"_"+decoy_name+"_TS"+no+".npy" 
    print(label_name,target_values[i])
    np.save(label_name,target_values[i])
