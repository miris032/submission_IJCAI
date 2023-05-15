import os.path

import numpy as np
import pandas as pd


def load_data(dataset_name, label_file, data_folder):
    if os.path.exists(f'{data_folder}/{dataset_name}.txt'):
        data = np.loadtxt(f'{data_folder}/{dataset_name}.txt', delimiter=",")
    elif os.path.exists(f'{data_folder}/{dataset_name}.csv'):
        data = np.asarray(pd.read_csv(f'{data_folder}/{dataset_name}.csv', header=None))
    else:
        raise FileNotFoundError(f'{data_folder}/{dataset_name}.*')

    assert os.path.exists(f'{data_folder}/{label_file}'), f'Label file does not exist: {label_file}'
    label = np.loadtxt(f'{data_folder}/{label_file}', delimiter=',')
    df_data = pd.DataFrame(data)
    df_label = pd.DataFrame(label)
    df = pd.concat((df_data, df_label), axis=1)

    return df
