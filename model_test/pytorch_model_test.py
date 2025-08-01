import os
import torch
import random
import typing
import datetime

import pandas as pd
import numpy as np

from . import abstract_model_test

from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryROC,
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
    MulticlassAUROC,
    MulticlassConfusionMatrix,
    MulticlassROC
)

from custom_metrics import (
    timing,
    storage
)

class PytorchModelTest(abstract_model_test.AbstractModelTest):
    def __init__(self, model, model_specs_dict: typing.Dict, output_path: str):
        self._model = model
        self._labeling_schema = model_specs_dict['feat_gen']['config']['labeling_schema']
        self._model_name = model_specs_dict['model_specs']['model_name']
        self._presaved_models_state_dict = model_specs_dict['model_specs']['presaved_paths']
        self._model_specs_dict = model_specs_dict['model_specs']
        self._evaluation_metrics = []
        self._batch_size = model_specs_dict['model_specs']['hyperparameters']['batch_size']
        self._number_of_outputs = model_specs_dict['model_specs']['hyperparameters'].get('num_outputs', -1)
        # self._forward_output_path = model_specs_dict['model_specs']['paths']['forward_output_path']
        self._confusion_matrix = None
        self._roc_metrics = None

        self._run_id = f"{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}_pytorch_test"

        # TODO: Get this from json config file (DONE)
        art_path = os.path.dirname(output_path)
        self._artifacts_path = art_path

        if not os.path.exists(self._artifacts_path):
            os.makedirs(self._artifacts_path)
            print("Artifacts output directory created successfully")

        self._metrics_output_path = f"{self._artifacts_path}/metrics"
        if not os.path.exists(self._metrics_output_path):
            os.makedirs(self._metrics_output_path)
            print("Metrics output directory created successfully")


    def __seed_all(self, seed):
        # Reference
        # https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097
        if not seed:
            seed = 10

        print("[ Using Seed : ", seed, " ]")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    def __seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)


    def __model_cnn_forward(self, device, testloader, fold):
        self._model.eval()

        print(">> Executing forward to export")

        N_OF_ENTRIES = len(testloader) * self._batch_size
        # Number of input features for the flatten layer

        # store_tensor = torch.zeros((N_OF_ENTRIES, N_OF_FEATURES), dtype=torch.float32)
        print(f"store_tensor.shape = {store_tensor.shape}")

        print(">> Preallocated output tensor")
        store_tensor_index = 0

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                output = self._model.fc1_forward(data)
                output = output.detach()

                start_index = store_tensor_index
                end_index = store_tensor_index + self._batch_size
                for index in range(0, self._batch_size):
                    # isso aqui pode ta copiando a referencia
                    # e mantendo os valores sempres iguais
                    store_tensor[start_index + index] = output[index].clone()

                store_tensor_index = store_tensor_index + self._batch_size

        store_tensor = store_tensor.cpu().numpy()

        np.savez(f"{self._forward_output_path}/sample_model_fold_{fold}_fc1_forward.npz", store_tensor)


    def __test_model(self, device, testloader, fold):
        self._model.eval()

        if self._number_of_outputs > 1:
            accuracy_metric = MulticlassAccuracy(num_classes=self._number_of_outputs).to(device)
            f1_score_metric = MulticlassF1Score(num_classes=self._number_of_outputs).to(device)
            auc_roc_metric = MulticlassAUROC(num_classes=self._number_of_outputs).to(device)
            precision_score = MulticlassPrecision(num_classes=self._number_of_outputs).to(device)
            recall_score = MulticlassRecall(num_classes=self._number_of_outputs).to(device)
            confusion_matrix_metric = MulticlassConfusionMatrix(num_classes=self._number_of_outputs).to(device)
            roc_metric = MulticlassROC(num_classes=self._number_of_outputs, thresholds=1000).to(device)
        else:
            accuracy_metric = BinaryAccuracy().to(device)
            f1_score_metric = BinaryF1Score().to(device)
            auc_roc_metric = BinaryAUROC().to(device)
            precision_score = BinaryPrecision().to(device)
            recall_score = BinaryRecall().to(device)
            confusion_matrix_metric = BinaryConfusionMatrix().to(device)
            roc_metric = BinaryROC(thresholds=1000).to(device)

        # TODO: alterar para prealocar y_pred e y_true pra exeutar mais rapido
        # n_entries = self._batch_size * len(testloader)
        # y_pred = torch.zeros((n_entries, self._number_of_outputs)).to(device)
        # y_true = torch.zeros((n_entries, self._number_of_outputs)).to(device)
        y_pred = torch.tensor([]).to(device)
        y_true = torch.tensor([]).to(device)
        initial_entry = 0

        with torch.no_grad():
            for data, target in testloader:
                data = data.float()
                if len(target.shape) == 1:
                    target = target.reshape(-1, 1)
                target = target.float()

                output = self._model(data)
                output = output.detach()

                y_pred = torch.cat((y_pred, output))
                y_true = torch.cat((y_true, target))

                # for index in range(self._batch_size):
                #     y_pred[initial_entry + index] = output[index].clone()
                #     y_true[initial_entry + index] = target[index].clone()
                # initial_entry = initial_entry + self._batch_size

                accuracy_metric.update(output, target)
                f1_score_metric.update(output, target)
                # TODO: Find a better way to perform this computation
                if self._number_of_outputs == 6:
                    auc_roc_metric.update(output, torch.argmax(target, dim=1))
                else:
                    auc_roc_metric.update(output, target)
                precision_score.update(output, target)
                recall_score.update(output, target)

            # Calculate metrics
            acc = accuracy_metric.compute().cpu().numpy()
            f1 = f1_score_metric.compute().cpu().numpy()
            roc_auc = auc_roc_metric.compute().cpu().numpy()
            prec = precision_score.compute().cpu().numpy()
            recall = recall_score.compute().cpu().numpy()

            # Reshape y_pred and y_true to compute confusion matrix
            # TODO: Esse reshape deve ocorrer apenas se for necessário
            y_pred_conf_matrix = y_pred
            y_true_conf_matrix = y_true
            if self._number_of_outputs == 6:
                y_pred_conf_matrix = torch.argmax(y_pred, dim=1)
                y_true_conf_matrix = torch.argmax(y_true, dim=1)
            confusion_matrix = confusion_matrix_metric(y_pred_conf_matrix, y_true_conf_matrix)

            # TODO: encontrar uma forma melhor de fazer esse reshape
            y_true_roc = y_true.to(torch.int32)
            if self._number_of_outputs == 6:
                y_true_roc = torch.argmax(y_true_roc, dim=1)
            fpr, tpr, thresholds = roc_metric(y_pred, y_true_roc)

            if self._number_of_outputs == 6:
                self._fpr_multiclass = fpr.T.cpu().numpy()
                self._tpr_multiclass = tpr.T.cpu().numpy()
                self._thresholds_multiclass = thresholds.T.cpu().numpy()
            else:
                roc_metrics = torch.cat((fpr.reshape(-1, 1), tpr.reshape(-1, 1), thresholds.reshape(-1, 1)), dim=1)
                self._roc_metrics = roc_metrics.cpu().numpy()

            # TODO: Get the data format using the data
            dummy_input = torch.randn(64, 1, 44, 116, dtype=torch.float).to(device)
            if device.type == "cpu":
                print("detection time in cpu")
                timing_func = timing.pytorch_inference_time_cpu
            else:
                print("detection time in gpu")
                timing_func = timing.pytorch_inference_time_gpu
            inference_time = timing_func(self._model, dummy_input)
            inference_time = inference_time / len(dummy_input)

            # TODO: Change this to be only used in case model is random forest
            # model_size = storage.pytorch_compute_model_size_mb(self._model)

            # Append metrics on list
            self._evaluation_metrics.append([fold, acc, prec, recall, f1, roc_auc, inference_time])
            self._confusion_matrix = confusion_matrix.cpu().numpy()


    def execute(self, data):
        def collate_gpu(batch):
            x, t = torch.utils.data.dataloader.default_collate(batch)
            return x.to(device="cuda:0"), t.to(device="cuda:0")
        # Reset all seed to ensure reproducibility
        self.__seed_all(0)
        g = torch.Generator()
        g.manual_seed(42)

        # Use gpu to train as preference
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        fold_index = None

        # for fold_index in self._presaved_models_state_dict.keys():
        print('------------fold no---------{}----------------------'.format(fold_index))

        testloader = torch.utils.data.DataLoader(
                    data,
                    batch_size=self._batch_size,
                    generator=g,
                    worker_init_fn=self.__seed_worker,
                    collate_fn=collate_gpu)

        print(f"len(testloader) = {len(testloader)}")

        for presaved_key in self._presaved_models_state_dict.keys():
            fold_index = presaved_key
            self._model.load_state_dict(torch.load(self._presaved_models_state_dict[presaved_key], map_location='cpu'))

            self._model.to(device)

            # This is only used in case you want to generate data for random forest models
            # self.__model_cnn_forward(device, testloader, fold_index)

            # Perform test step
            self.__test_model(device, testloader, fold_index)

            # Export metrics
            metrics_df = pd.DataFrame(self._evaluation_metrics, columns=["fold", "acc", "prec", "recall", "f1", "roc_auc", "inference_time"])
            metrics_df.to_csv(f"{self._metrics_output_path}/test_metrics_{self._labeling_schema}_{self._model_name}_BS{self._batch_size}_fold_{fold_index}.csv")
            confusion_matrix_df = pd.DataFrame(self._confusion_matrix)
            confusion_matrix_df.to_csv(f"{self._metrics_output_path}/confusion_matrix_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")

            if self._number_of_outputs == 6:
                tpr_df = pd.DataFrame(self._tpr_multiclass)
                fpr_df = pd.DataFrame(self._fpr_multiclass)
                thresholds_df = pd.DataFrame(self._thresholds_multiclass)

                tpr_df.to_csv(f"{self._metrics_output_path}/tpr_multiclass_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
                fpr_df.to_csv(f"{self._metrics_output_path}/fpr_multiclass_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
                thresholds_df.to_csv(f"{self._metrics_output_path}/thresholds_multiclass_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
            else:
                roc_metrics_df = pd.DataFrame(self._roc_metrics, columns=["fpr", "tpr", "thresholds"])
                roc_metrics_df.to_csv(f"{self._metrics_output_path}/roc_metrics_{self._labeling_schema}_fold_{fold_index}_{self._model_name}.csv")
