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

# Model train
python3 $HOME/Intrusion-Detection/execute_model_train_validation.py --model_train_valid_config config_jsons/TOW/model_train/RF_train_val_config.json

# Model test
python3 $HOME/Intrusion-Detection/execute_model_test.py --model_test_config config_jsons/TOW/model_test/RF_test_config.json

# Metrics extraction
python3 $HOME/Intrusion-Detection/plot_metrics.py --path output/TOW/RF
