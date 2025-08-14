# Ambient activate
source $HOME/Intrusion-Detection/venv/bin/activate

# RF Features - AIED
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/AIED/feature_generation/AVTP_RandomForest_train.json
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/AIED/feature_generation/AVTP_RandomForest_test.json

# CNN Features - AIED
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/AIED/feature_generation/AVTP_CNNIDS_train.json
python3 $HOME/Intrusion-Detection/scripts/execute_feature_generator.py --feat_gen_config config_jsons/AIED/feature_generation/AVTP_CNNIDS_test.json
