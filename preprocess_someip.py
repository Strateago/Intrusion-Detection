import argparse
import pandas as pd
import numpy as np
import os

#TODO: Change the test shape to become a 2D array (Not One Hot encoded)
def main():
    parser = argparse.ArgumentParser(description="Preprocess the SOMEIP dataset")
    parser.add_argument("--path", type=str, required=True, help="Base path where the dataset is located.")
    args = parser.parse_args()
    path = args.path

    os.makedirs(f'{path}/Preprocessed', exist_ok=True)
    print("Start aggregating X")
    for file in os.listdir(path):
        if not os.path.isfile(f'{path}/{file}'):
            continue
        data = pd.read_pickle(f'{path}/{file}')
        if 'X' in file:
            data = np.sum(data, axis=1)
        else:
            print(np.unique(data, return_counts=True))
        name = file.split('.')[0]
        print(name, ' ', data.shape)
        np.save(f'{path}/Preprocessed/{name}', data)

    print('Done')

if __name__ == "__main__":
    main()
