# officehome_uncertainty

## How to run
There are 12 scripts in run_train.sh
Each of them corresponds to one task with a single source domain and a target domain
### Steps
1. Change the 3 path in the script: data_dir, save_dir and resnet_model_dir to: 
- data_dir=which the dataset folder is placed
- save_dir=in which folder you like the training log file and models is saved
- resnet_model_dir=path to the downloaded pytorch ImageNet model
2. Set your cuda device id in the script:
- device='0' for example
3. Run
