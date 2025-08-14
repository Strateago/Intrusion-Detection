import argparse
import pandas as pd
import numpy as np
import os
import numpy as np

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
            # Adjusting the shape of the data
            data = data.reshape(-1, data.shape[-1])
        elif 'test' in file:
            aux = []
            for i in range(len(data)):
                if data[i][0] == 1:
                    aux.append([0])
                elif data[i][1] == 1:
                    aux.append([1])
                elif data[i][2] == 1:
                    aux.append([2])
                elif data[i][3] == 1:
                    aux.append([3])
                else:
                    aux.append([4])
            data = np.array(aux)
        if 'Y' in file:
            # Repeat the labels of each group to the 60 packets of the group
            data = np.repeat(data, 60)
            print(np.unique(data, return_counts=True))

        name = file.split('.')[0]
        print(name, ' ', data.shape)
        np.save(f'{path}/Preprocessed/{name}', data)

    print('Done')

if __name__ == "__main__":
    main()