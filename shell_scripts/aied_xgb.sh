# Ambient activate
source $HOME/Intrusion-Detection/venv/bin/activate

# Model train
python3 $HOME/Intrusion-Detection/scripts/execute_model_train_validation.py --model_train_valid_config config_jsons/AIED/model_train/AVTP_XGBOOST_train.json

# Model test
python3 $HOME/Intrusion-Detection/scripts/execute_model_test.py --model_test_config config_jsons/AIED/model_test/AVTP_XGBOOST_test.json

# Metrics extraction
python3 $HOME/Intrusion-Detection/scripts/plot_metrics.py --path output/AIED/XGB
