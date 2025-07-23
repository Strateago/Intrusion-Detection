#!/bin/bash
#SBATCH --job-name=RF
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --mem 16G
#SBATCH -c 8
#SBATCH -o logs/RF/job.log
#SBATCH --output=logs/RF/job_output.txt
#SBATCH --error=logs/RF/job_error.txt

#carregar versão python
module load Python/3.10
#ativar ambiente
source $HOME/automotive-ids-evaluation-framework/venv/bin/activate
#extrair features
python3 $HOME/automotive-ids-evaluation-framework/execute_feature_generator.py --feat_gen_config created_jsons/RF_train_feat_gen_config.json
python3 $HOME/automotive-ids-evaluation-framework/execute_feature_generator.py --feat_gen_config created_jsons/RF_test_feat_gen_config.json
#executar o treino
python3 $HOME/automotive-ids-evaluation-framework/execute_model_train_validation.py --model_train_valid_config created_jsons/RF_train_val_config.json
#executar o teste
python3 $HOME/automotive-ids-evaluation-framework/execute_model_test.py --model_test_config created_jsons/RF_test_config.json
#criar um diretório de resultados para o modelo selecionado
mkdir -p $HOME/automotive-ids-evaluation-framework/output/RF/results
#salvar as imagens
python3 $HOME/automotive-ids-evaluation-framework/output/plot_conf_matrix.py --path RF
