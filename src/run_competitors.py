import os.path
import warnings

import pandas as pd
from competitors.util import load_data
from competitors.hdddm import HDDDM_Competitor
from competitors.pcacd import PCACD_Competitor
from competitors.kdqtree import KdqTree_Competitor
from competitors.adwin import Adwin_Competitor

competitor_func_mapping = {'hdddm': '_run_HDDDM_competitor',
                           'adwin': '_run_ADWIN_competitor',
                           'kdqtree': '_run_KdqTree_competitor',
                           'pcacd': '_run_PCACD_competitor'}

hdddm_batch_size = {'Cancer': 10,
                    'Synthetic': 200,
                    'LED_10000_10': 200,
                    'KDD_all': 200,
                    'MSL': 200,
                    'Poker': 200,}
seed = 42


def _run_HDDDM_competitor(window_length, dataset):
    hdddm = HDDDM_Competitor(batch_size=window_length * 2,
                             dataset=dataset,
                             seed=seed)
    hdddm_drift_prediction = hdddm.run()
    return hdddm_drift_prediction


def _run_PCACD_competitor(window_length, dataset):
    pcacd = PCACD_Competitor(dataset=dataset,
                             window_length=window_length)
    pcacd_drift_prediction = pcacd.run()
    return pcacd_drift_prediction


def _run_KdqTree_competitor(window_length, dataset):
    kdqTree = KdqTree_Competitor(dataset=dataset,
                                 window_length=window_length,
                                 alpha=0.05,
                                 bootstrap_samples=1000,
                                 count_ubound=100)
    kdq_drift_prediction = kdqTree.run()
    return kdq_drift_prediction


def _run_ADWIN_competitor(window_length, dataset):
    adwin = Adwin_Competitor(dataset=dataset)
    adwin_drift_prediction = adwin.run()

    return adwin_drift_prediction


def run_competitors(ts_data_name, models, label_file, window_length, exp_folder, data_folder):
    output_file = f'{exp_folder}/competitor_result_{ts_data_name}.csv'

    if os.path.exists(output_file):
        warnings.warn(f'Competitor result already exists: {output_file}')
        return
    if len(models) == 0:
        return

    print(f'Competitors on dataset {ts_data_name}...')
    df = load_data(ts_data_name, label_file, data_folder)
    results = []
    final_result = []

    for model in models:
        drift_prediction = eval(f'{competitor_func_mapping[model]}')(window_length, df)
        results.append(pd.Series(drift_prediction))

    pred = pd.concat(results, axis=1)
    win_len_sq = pd.Series([window_length for _ in range(pred.shape[0])])
    seed_sq = pd.Series([seed for _ in range(pred.shape[0])])
    dataset_sq = pd.Series([ts_data_name for _ in range(pred.shape[0])])
    label_sq = df.iloc[:, -1]

    combination = pd.concat([dataset_sq, win_len_sq, seed_sq, pred, label_sq], axis=1)
    final_result.append(combination)

    final_result_df = pd.concat(final_result, axis=0)
    final_result_df.to_csv(output_file, index=False,
                           header=['dataset', 'win_len', 'seed'] + models + ['label'])