#!/bin/bash
#SBATCH --job-name=PruCNN
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -o logs/CNN/job.log
#SBATCH --output=logs/CNN/job_output.txt
#SBATCH --error=logs/CNN/job_error.txt

# Ambient activate
source $HOME/Intrusion-Detection/venv/bin/activate

# Model train
python3 $HOME/Intrusion-Detection/execute_model_train_validation.py --model_train_valid_config config_jsons/AIED/model_train/AVTP_RandomForest_test.json

# Model test
python3 $HOME/Intrusion-Detection/execute_model_test.py --model_test_config config_jsons/AIED/model_test/AVTP_RandomForest_test.json

# Metrics extraction
python3 $HOME/Intrusion-Detection/plot_metrics.py --path CNN
