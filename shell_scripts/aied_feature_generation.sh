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

# RF Features - AIED
python3 $HOME/Intrusion-Detection/execute_feature_generator.py --feat_gen_config config_jsons/AIED/feature_generation/AVTP_RandomForest_train.json
python3 $HOME/Intrusion-Detection/execute_feature_generator.py --feat_gen_config config_jsons/AIED/feature_generation/AVTP_RandomForest_test.json

# CNN Features - AIED
python3 $HOME/Intrusion-Detection/execute_feature_generator.py --feat_gen_config config_jsons/AIED/feature_generation/AVTP_CNNIDS_train.json
python3 $HOME/Intrusion-Detection/execute_feature_generator.py --feat_gen_config config_jsons/AIED/feature_generation/AVTP_CNNIDS_test.json
