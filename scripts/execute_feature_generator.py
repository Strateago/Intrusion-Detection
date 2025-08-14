import argparse
import json
import os

from feature_generator import cnn_ids_feature_generator


class FeatureGenerationExecutor:
    AVAILABLE_FEATURE_GENERATORS = {
        "CNNIDSFeatureGenerator": cnn_ids_feature_generator.CNNIDSFeatureGenerator
    }

    def __init__(self, config_path: str, benchmark: bool = False):
        self.config_path = config_path
        self.benchmark = benchmark
        self.config_dict = {}
        self.feature_generator_instance = None

    def load_config(self):
        try:
            with open(self.config_path, 'r') as config_file:
                self.config_dict = json.load(config_file)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"load_config: Error: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"load_config: Error decoding JSON: {e}")

        print("##### Loaded configuration dictionary #####")
        print(json.dumps(self.config_dict, indent=4, sort_keys=True))

    def prepare_output_directory(self):
        output_path = self.config_dict["paths"]["output_path"]
        os.makedirs(output_path, exist_ok=True)

    def initialize_feature_generator(self):
        feature_generator_name = self.config_dict['feature_generator']
        feature_generator_config = self.config_dict['config']

        if feature_generator_name not in self.AVAILABLE_FEATURE_GENERATORS:
            raise KeyError(f"Selected feature generator '{feature_generator_name}' is NOT available!")

        generator_class = self.AVAILABLE_FEATURE_GENERATORS[feature_generator_name]
        self.feature_generator_instance = generator_class(feature_generator_config)

        print(f"> Selected feature generator: {feature_generator_name}")

    def execute(self):
        if self.benchmark:
            print("> Execution time benchmark generation")
            self.feature_generator_instance.benchmark_execution_time()
        else:
            print("> Generating features...")
            self.feature_generator_instance.generate_features(self.config_dict["paths"])

        print("Feature generator successfully executed!")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Execute feature generation step')
    parser.add_argument('--feat_gen_config', required=True,
                        help='JSON File containing the configs for the specified feature generation method')
    parser.add_argument('--bench_time', action='store_true',
                        help='Flag to execute the feature generator execution time benchmark')
    return parser.parse_args()


def main():
    args = parse_arguments()

    executor = FeatureGenerationExecutor(
        config_path=args.feat_gen_config,
        benchmark=args.bench_time
    )

    executor.load_config()
    executor.prepare_output_directory()
    executor.initialize_feature_generator()
    executor.execute()


if __name__ == "__main__":
    main()
