import pandas as pd
import os
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parent.parent


def load_data(data):
    path = f'{root}/data/' + data

    if os.path.exists(path + '.pkl'):
        _mydata = pd.read_pickle(path + '.pkl')

    elif os.path.exists(path + '.txt'):
        _mydata = np.loadtxt(path + '.txt', delimiter=",")

    elif os.path.exists(path + '.csv'):
        _mydata = np.asarray(pd.read_csv(path + '.csv', delimiter=","))
    #         _mydata = _mydata)
    else:
        raise FileNotFoundError(path+'.*')

    return _mydata


def load_data_CD():
    _data = 'concept_drifted_data'
    _mydata = load_data(_data)

    ran = np.arange(0.05, 1.05, 0.05)

    col_to_drop = []
    for i in range(len(_mydata.columns)):
        if len(_mydata[_mydata.columns[i]].unique()) == 10000:
            name_col = _mydata.columns[i]
            x = _mydata[name_col]
            binned = np.digitize(x, ran)
            col_to_drop.append(name_col)
            if i % 2 == 1:
                _mydata[f'{name_col}_binned'] = binned

    print(col_to_drop)

    _mydata = _mydata.drop(col_to_drop, axis=1).to_numpy()

    return _mydata
