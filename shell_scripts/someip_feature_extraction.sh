#!/bin/bash
#SBATCH --job-name=RF
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -o logs/RF/job.log
#SBATCH --output=logs/RF/job_output.txt
#SBATCH --error=logs/RF/job_error.txt

# Ambient activate
source $HOME/Intrusion-Detection/venv/bin/activate

# Preprocessing - SOMEIP
python3 $HOME/Intrusion-Detection/scripts/preprocess_someip.py --path datasets/SOMEIP_IDS

# RF Features - SOMEIP
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/SOMEIP/feature_generation/RF_train_feat_gen_config.json
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/SOMEIP/feature_generation/RF_test_feat_gen_config.json

# CNN Features - SOMEIP
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/SOMEIP/feature_generation/CNN_train_feat_gen_config.json
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/SOMEIP/feature_generation/CNN_test_feat_gen_config.json
