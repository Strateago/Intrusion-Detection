import gc
import pandas as pd
import numpy as np
import typing
import time
from scapy.all import *

from . import abstract_feature_generator
from . import labeling_schemas

from sklearn.preprocessing import OneHotEncoder

DEFAULT_WINDOW_SIZE = 44
DEFAULT_NUMBER_OF_BYTES = 58
DEFAULT_WINDOW_SLIDE = 1
AVTP_PACKETS_LENGHT = 438
DEFAULT_LABELING_SCHEMA = "AVTP_Intrusion_dataset"
DEFAULT_DATASET = "AVTP_Intrusion"
DEFAULT_SUM_X = False

LABELING_SCHEMA_FACTORY = {
    "AVTP_Intrusion_dataset": labeling_schemas.avtp_intrusion_labeling_schema,
    "TOW_IDS_dataset_one_class": labeling_schemas.tow_ids_one_class_labeling_schema,
    "TOW_IDS_dataset_multi_class": labeling_schemas.tow_ids_multi_class_labeling_schema
}

class CNNIDSFeatureGenerator(abstract_feature_generator.AbstractFeatureGenerator):
    def __init__(self, config: typing.Dict):
        self._window_size = config.get('window_size', DEFAULT_WINDOW_SIZE)
        self._number_of_bytes = config.get('number_of_bytes', DEFAULT_NUMBER_OF_BYTES)
        self._window_slide = config.get('window_slide', DEFAULT_WINDOW_SLIDE)
        self._number_of_columns = self._number_of_bytes * 2
        self._labeling_schema = config.get('labeling_schema', DEFAULT_LABELING_SCHEMA)
        self._sum_x = config.get('sum_x', DEFAULT_SUM_X)
        self._data_suffix = config.get('suffix')

        self._multiclass = config.get('multiclass', False)

        self._dataset = config.get('dataset', DEFAULT_DATASET)

        self._output_path_suffix = f"{self._labeling_schema}_Wsize_{self._window_size}_Cols_{self._number_of_columns}_Wslide_{self._window_slide}_MC_{self._multiclass}_sumX_{self._sum_x}"

        self._filter_avtp_packets = True if (self._labeling_schema) == "AVTP_Intrusion_dataset" else False
        print(f"filter_avtp_packets = {self._filter_avtp_packets}")

    def generate_features(self, paths_dictionary: typing.Dict):
        CNN_IDS_FEAT_GEN_AVAILABLE_DATASETS = {
            "AVTP_Intrusion_dataset": self.__avtp_dataset_generate_features,
            "TOW_IDS_dataset": self.__tow_ids_dataset_generate_features
        }

        if self._dataset not in CNN_IDS_FEAT_GEN_AVAILABLE_DATASETS:
            raise KeyError(f"Selected dataset: {self._dataset} is NOT available for CNN IDS Feature Generator!")

        feature_generator = CNN_IDS_FEAT_GEN_AVAILABLE_DATASETS[self._dataset](paths_dictionary)

    def __tow_ids_dataset_generate_features(self, paths_dictionary: typing.Dict):
        # Load raw packets
        labels = pd.read_csv(paths_dictionary["y_train_path"], header=None, names=["index", "Class", "Description"])
        labels = labels.drop(columns=["index"])
        converted_packets_list = []
        raw_packets = rdpcap(paths_dictionary["training_packets_path"])

        print(">> Loading raw packets...")
        for raw_packet in raw_packets:
            converted_packet = np.frombuffer(raw(raw_packet), dtype='uint8')

            converted_packet_len = len(converted_packet)
            if converted_packet_len < self._number_of_bytes:
                bytes_to_pad = self._number_of_bytes - converted_packet_len
                converted_packet = np.pad(converted_packet, (0, bytes_to_pad), 'constant')
            else:
                converted_packet = converted_packet[0:self._number_of_bytes]

            converted_packets_list.append(converted_packet)

        converted_packets = np.array(converted_packets_list, dtype='uint8')

        # Preprocess packets
        print(">> Preprocessing raw packets...")
        preprocessed_packets = self.__preprocess_raw_packets(converted_packets, split_into_nibbles=True)

        # Aggregate features and labels
        print(">> Aggregating and labeling...")
        aggregated_X, aggregated_y = self.__aggregate_based_on_window_size(preprocessed_packets, labels)

        np.savez(f"{paths_dictionary['output_path']}/X_{self._data_suffix}_{self._output_path_suffix}", aggregated_X)

        y_df = pd.DataFrame(aggregated_y, columns=["Class"])
        y_df.to_csv(f"{paths_dictionary['output_path']}/y_{self._data_suffix}_{self._output_path_suffix}.csv")

    def __avtp_dataset_generate_features(self, paths_dictionary: typing.Dict):
        raw_injected_only_packets = self.__read_raw_packets(paths_dictionary['injected_only_frame_path'])
        injected_only_packets_array = self.__convert_raw_packets(raw_injected_only_packets)

        if self._sum_x:
            X = np.empty(shape=(0, self._number_of_columns), dtype='uint8')
        else:
            X = np.empty(shape=(0, self._window_size, self._number_of_columns), dtype='uint8')
        y = np.array([], dtype='uint8')

        for injected_raw_packets_path in paths_dictionary['injected_data_paths']:
            # Load raw packets
            raw_packets = self.__read_raw_packets(injected_raw_packets_path)

            # Convert loaded packets to np array with uint8_t size
            packets_array = self.__convert_raw_packets(raw_packets)

            # Preprocess packets
            preprocessed_packets = self.__preprocess_raw_packets(packets_array, split_into_nibbles=True)

            # Generate labels
            labels = self.__generate_labels(packets_array, injected_only_packets_array)

            # Aggregate features and labels
            aggregated_X, aggregated_y = self.__aggregate_based_on_window_size(preprocessed_packets, labels)
            aggregated_y = np.array(aggregated_y, dtype='uint8')

            # Concatenate both indoors injected packets
            X = np.concatenate((X, aggregated_X), axis=0, dtype='uint8')
            y = np.concatenate((y, aggregated_y), axis=0, dtype='uint8')

            np.savez(f"{paths_dictionary['output_path']}/X_{self._data_suffix}_{self._output_path_suffix}", X)
            np.savez(f"{paths_dictionary['output_path']}/y_{self._data_suffix}_{self._output_path_suffix}", y)


    def load_features(self, paths_dictionary: typing.Dict):
        X = np.load(paths_dictionary['X_path'])
        X = X.f.arr_0
        if self._sum_x == False:
            X = X.reshape((X.shape[0], -1, self._window_size, self._number_of_columns))
        print(f"shape X = {X.shape}")

        if (self._dataset == "TOW_IDS_dataset"):
            y = pd.read_csv(paths_dictionary['y_path'])
            y = y.drop(columns=["Unnamed: 0"])
            if self._dataset == "TOW_IDS_multiclass":
                y["Class"] = y["Class"].map(
                    {
                        "Normal": 0,
                        "C_D": 1,
                        "C_R": 2,
                        "M_F": 3,
                        "P_I": 4,
                        "F_I": 5
                    }
                )
            y = np.array(y["Class"].values)
        else:
            y = np.load(paths_dictionary['y_path'])
            y = y.f.arr_0

        print(f"shape Y = {y.shape}")

        if (self._multiclass):
            y = y.reshape(-1, 1)
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False).fit(y)
            y = ohe.transform(y)

        return [[X[i], y[i]] for i in range(X.shape[0])]

    def __read_raw_packets(self, pcap_filepath):
        raw_packets = rdpcap(pcap_filepath)

        raw_packets_list = []

        for packet in raw_packets:
            if self._filter_avtp_packets:
                if (len(packet) == AVTP_PACKETS_LENGHT):
                    raw_packets_list.append(raw(packet))
            else:
                raw_packets_list.append(raw(packet))

        return raw_packets_list


    def __convert_raw_packets(self, raw_packets_list, zero_padding=False):
        converted_packets_list = []

        for raw_packet in raw_packets_list:
            converted_packet = np.frombuffer(raw_packet, dtype='uint8')

            if zero_padding:
                converted_packet_len = len(converted_packet)
                if converted_packet_len < self._number_of_bytes:
                    bytes_to_pad = self._number_of_bytes - converted_packet_len
                    converted_packet = np.pad(converted_packet, (0, bytes_to_pad), 'constant')
                else:
                    converted_packet = converted_packet[0:self._number_of_bytes]

            converted_packets_list.append(converted_packet)

        return np.array(converted_packets_list, dtype='uint8')


    def __is_array_in_list_of_arrays(self, array_to_check, list_np_arrays):
        # Reference:
        # https://stackoverflow.com/questions/23979146/check-if-numpy-array-is-in-list-of-numpy-arrays
        is_in_list = np.any(np.all(array_to_check == list_np_arrays, axis=1))

        return is_in_list


    def __generate_labels(self, packets_list, injected_packets):
        labels_list = []

        for packet in packets_list:
            current_label = 0

            if self.__is_array_in_list_of_arrays(packet, injected_packets):
                current_label = 1

            labels_list.append(current_label)

        return labels_list


    def __select_packets_bytes(self, packets_list):
        selected_packets = packets_list[:, 0:self._number_of_bytes]

        return np.array(selected_packets, dtype='uint8')


    def __calculate_difference_module(self, selected_packets):
        difference_array = np.diff(selected_packets, axis=0)
        difference_module = np.mod(difference_array, 256)

        return difference_module


    def __split_byte_into_nibbles(self, byte):
        high_nibble = (byte >> 4) & 0xf
        low_nibble = (byte) & 0xf

        return high_nibble, low_nibble


    def __create_nibbles_matrix(self, difference_module):
        nibbles_matrix = []

        # Difference matrix has "n" rows and "p" columns
        for row_index in range(len(difference_module)):
            nibbles_row = []
            for column_index in range(len(difference_module[row_index])):
                hi_ni, low_ni = self.__split_byte_into_nibbles(difference_module[row_index, column_index])

                nibbles_row.append(hi_ni)
                nibbles_row.append(low_ni)

            nibbles_matrix.append(np.array(nibbles_row, dtype='uint8'))

        return np.array(nibbles_matrix, dtype='uint8')

    def __split_into_nibbles(self, x1):
        # Ensure the dtype is large enough to hold the shifted values without overflow
        x1_np = x1.astype(np.uint8)

        # Prepare a mask to isolate nibbles. 0xF is 1111 in binary, which isolates a nibble.
        mask = 0xF

        # Extract nibbles.
        # The idea is to shift the original numbers right by 0 and 4 bits
        # and then mask off the lower 4 bits.
        nibbles = np.zeros((x1_np.shape[0], x1_np.shape[1] * 2), dtype=np.uint8)

        for i in range(0, 2):
            nibbles[:, int(not i)::2] = (x1_np >> (i * 4)) & mask

        return nibbles


    def __preprocess_raw_packets(self, converted_packets, split_into_nibbles=True):
        # Select first 58 bytes
        selected_packets = self.__select_packets_bytes(converted_packets)

        # Calculate difference and module between rows
        diff_module_packets = self.__calculate_difference_module(selected_packets)

        # Split difference into two nibbles
        if split_into_nibbles:
            # diff_module_packets = self.__create_nibbles_matrix(diff_module_packets)
            diff_module_packets = self.__split_into_nibbles(diff_module_packets)

        return diff_module_packets


    def __aggregate_based_on_window_size(self, x_data, y_data):
        # Prepare the list for the transformed data
        X, y = list(), list()

        # Loop of the entire data set
        for i in range(x_data.shape[0]):
            # Compute a new (sliding window) index
            start_ix = i*(self._window_slide)
            end_ix = start_ix + self._window_size - 1 + 1

            # If index is larger than the size of the dataset, we stop
            if end_ix >= x_data.shape[0]:
                break

            # Get a sequence of data for x
            seq_X = x_data[start_ix:end_ix]
            if self._sum_x:
                seq_X = np.sum(seq_X, axis=0)

            # Get a squence of data for y
            tmp_seq_y = y_data[start_ix : end_ix]

            # Labeling schema
            seq_y = LABELING_SCHEMA_FACTORY[self._labeling_schema](tmp_seq_y)

            # Append the list with sequences
            X.append(seq_X)
            y.append(seq_y)

        # Make final arrays
        x_array = np.array(X, dtype='uint8')
        # y_array = np.array(y, dtype='uint8')

        return x_array, y

    def benchmark_execution_time(self):
        print(">> Benchmarking feature generator execution time")
        n_iter = 10000
        execution_time_list = []
        # preprocess_time_list = []
        # agg_time_list = []
        # select_time_list = []
        # diff_time_list = []
        # nibble_time_list = []

        labels=0

        for i in range(n_iter):
            # Random input in the shape of the data
            # Regenerate at each iteration to avoid it to be in cache and bias the measurement
            random_packets = (255*np.random.rand(45, 56)).astype('uint8')
            # start_time = time.time()
            start_time = time.perf_counter_ns()
            # Preprocess packets
            preprocessed_packets = self.__preprocess_raw_packets(random_packets, split_into_nibbles=True)
            # Select first 58 bytes
            # selected_packets = self.__select_packets_bytes(random_packets)
            # select_bytes_time = time.perf_counter_ns()

            # Calculate difference and module between rows
            # diff_module_packets = self.__calculate_difference_module(selected_packets)
            # diff_module_time = time.perf_counter_ns()

            # Split difference into two nibbles
            # diff_module_packets = self.__split_into_nibbles(diff_module_packets)
            # preprocess_end = time.perf_counter_ns()

            # Aggregate features and labels
            aggregated_X, aggregated_y = self.__aggregate_based_on_window_size(preprocessed_packets, labels)
            # end_time = time.time()
            end_time = time.perf_counter_ns()
            elapsed_total_time = end_time - start_time
            # elapsed_agg_time = end_time - preprocess_end
            # elapsed_preprocess_time = preprocess_end - start_time
            # elapsed_select_time = select_bytes_time - start_time
            # elapsed_diff_time = diff_module_time - select_bytes_time
            # elapsed_nibble_time = preprocess_end - diff_module_time

            execution_time_list.append(elapsed_total_time)
            # preprocess_time_list.append(elapsed_preprocess_time)
            # agg_time_list.append(elapsed_agg_time)
            # select_time_list.append(elapsed_agg_time)
            # diff_time_list.append(elapsed_diff_time)
            # nibble_time_list.append(elapsed_nibble_time)

        # Compute the elapsed time
        mean_execution_time = np.mean(execution_time_list)
        std_execution_time = np.std(execution_time_list)

        # mean_preprocess_time = np.mean(preprocess_time_list)
        # std_preprocess_time = np.std(preprocess_time_list)

        # mean_agg_time = np.mean(agg_time_list)
        # std_agg_time = np.std(agg_time_list)

        # mean_select_time = np.mean(select_time_list)
        # std_select_time = np.std(select_time_list)

        # mean_diff_time = np.mean(diff_time_list)
        # std_diff_time = np.std(diff_time_list)

        # mean_nibble_time = np.mean(nibble_time_list)
        # std_nibble_time = np.std(nibble_time_list)

        print(f"Mean execution time: {mean_execution_time}, Std: {std_execution_time}")
        # print(f"Mean preprocess time: {mean_preprocess_time}, Std: {std_preprocess_time}")
        # print(f"Mean select time: {mean_select_time}, Std: {std_select_time}")
        # print(f"Mean diff time: {mean_diff_time}, Std: {std_diff_time}")
        # print(f"Mean nibble time: {mean_nibble_time}, Std: {std_nibble_time}")
        # print(f"Mean agg time: {mean_agg_time}, Std: {std_agg_time}")
