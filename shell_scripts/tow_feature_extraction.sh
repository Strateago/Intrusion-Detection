# Ambient activate
source $HOME/Intrusion-Detection/venv/bin/activate

# RF Features - TOW
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/TOW/feature_generation/RF_train_feat_gen_config.json
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/TOW/feature_generation/RF_test_feat_gen_config.json

# CNN Features - TOW
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/TOW/feature_generation/CNN_train_feat_gen_config.json
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/TOW/feature_generation/CNN_test_feat_gen_config.json
