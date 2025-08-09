# Ambient activate
source $HOME/Intrusion-Detection/venv/bin/activate

# Model train
python3 $HOME/Intrusion-Detection/scripts/execute_model_train_validation.py --model_train_valid_config config_jsons/TOW/model_train/CNN_train_val_config.json

# Model test
python3 $HOME/Intrusion-Detection/scripts/execute_model_test.py --model_test_config config_jsons/TOW/model_test/CNN_test_config.json

# Metrics extraction
python3 $HOME/Intrusion-Detection/scripts/plot_metrics.py --path output/TOW/classification
