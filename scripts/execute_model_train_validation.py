import argparse
import json

from feature_generator import cnn_ids_feature_generator
from models import (
    conv_net_ids,
    multiclass_conv_net_ids,
    pruned_conv_net_ids,
    sklearn_classifier,
)
from model_train_validation import (
    pytorch_model_train_validate,
    sklearn_model_train_validate
)

# Available feature generators
AVAILABLE_FEATURE_GENERATORS = {
    "CNNIDSFeatureGenerator": cnn_ids_feature_generator.CNNIDSFeatureGenerator
}

# Available models
AVAILABLE_IDS = {
    "CNNIDS": conv_net_ids.ConvNetIDS,
    "MultiClassCNNIDS": multiclass_conv_net_ids.MultiClassConvNetIDS,
    "PrunedCNNIDS": pruned_conv_net_ids.PrunedConvNetIDS,
    "SklearnClassifier": sklearn_classifier.SklearnClassifier,
}

# Available frameworks
AVAILABLE_FRAMEWORKS = {
    "pytorch": pytorch_model_train_validate.PytorchModelTrainValidation,
    "sklearn": sklearn_model_train_validate.SklearnModelTrainValidation
}


def main(args):
    # Load configuration file
    try:
        with open(args.model_train_valid_config, 'r') as config_file:
            config_dict = json.load(config_file)
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON: {e}")
        return

    print("##### Loaded Configuration File #####")
    print(json.dumps(config_dict, indent=4, sort_keys=True))

    # Extract configuration sections
    feat_gen_config = config_dict['feat_gen']
    model_specs = config_dict['model_specs']
    output_path = model_specs['paths']['models_output_path']

    # Validate feature generator
    feature_generator_name = feat_gen_config['feature_generator']
    if feature_generator_name not in AVAILABLE_FEATURE_GENERATORS:
        raise KeyError(f"Feature generator '{feature_generator_name}' is NOT available!")

    feature_generator_config = feat_gen_config['config']
    feature_generator_load_paths = feat_gen_config['load_paths']

    # Validate model
    model_name = model_specs['model']
    if model_name not in AVAILABLE_IDS:
        raise KeyError(f"Model '{model_name}' is NOT available!")

    # Validate framework
    framework = model_specs['framework']
    if framework not in AVAILABLE_FRAMEWORKS:
        raise KeyError(f"Framework '{framework}' is NOT available!")

    # Load features
    print("> Loading features...")
    selected_feature_generator = AVAILABLE_FEATURE_GENERATORS[feature_generator_name](feature_generator_config)
    data = selected_feature_generator.load_features(feature_generator_load_paths)

    # Create model
    print("> Creating model...")
    if framework == "pytorch":
        num_outputs = model_specs.get('hyperparameters', {}).get('num_outputs', 1)
        num_ensemble_inputs = model_specs.get('hyperparameters', {}).get('ensemble_inputs', 2)

        if model_name in ["CNNIDS", "PrunedCNNIDS", "MultiClassCNNIDS"]:
            if num_outputs > 1:
                model = AVAILABLE_IDS[model_name](number_of_outputs=num_outputs)
            else:
                model = AVAILABLE_IDS[model_name]()

        print(f">> {model_name} was created with {num_outputs} outputs")

    elif framework == "sklearn":
        model = AVAILABLE_IDS[model_name](model_specs)

    # Train and evaluate model
    print("> Initializing model training and evaluation...")
    trainer = AVAILABLE_FRAMEWORKS[framework](model, model_specs, output_path)
    trainer.execute(data)

    print("Model trained successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the model training and validation process')
    parser.add_argument(
        '--model_train_valid_config',
        required=True,
        help='Path to the JSON file containing model training and validation configuration'
    )
    args = parser.parse_args()
    main(args)
