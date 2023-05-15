import json
import os
import pickle

import pandas as pd
import numpy as np


def load_data_parameters(dataset, config_folder):
    data_config = json.load(open(os.path.join(config_folder, 'data_config.json')))

    window_length_list = data_config[dataset]['window_length']
    overlap_list = data_config[dataset]['overlap']
    subsets_bound_list = data_config[dataset]['subsets_bound']
    approximation_type = data_config[dataset]['approximation_type']
    concept_size_list = data_config[dataset]['concept_size'] if dataset in ['Synthetic', 'LED'] else [None]
    number_of_concepts_list = data_config[dataset]['number_of_concepts'] if dataset in ['Synthetic', 'LED'] else [None]
    categories_per_feature_list = data_config[dataset]['categories_per_feature'] if dataset == 'Synthetic' else [None]
    correlation_func_list = data_config[dataset]['correlation_func'] if dataset == 'Synthetic' else [None]

    return window_length_list, overlap_list, concept_size_list, number_of_concepts_list, categories_per_feature_list, \
           correlation_func_list, approximation_type, subsets_bound_list


def load_slidshaps_parameters(config_folder):
    model_config = json.load(open(os.path.join(config_folder, 'model_config.json')))

    statistical_test_list = model_config['slidshaps']['statistical_test']
    gamma_list = model_config['slidshaps']['gamma']
    buffer_size_list = model_config['slidshaps']['buffer_size']
    alpha_list = model_config['slidshaps']['alpha']

    return statistical_test_list, gamma_list, buffer_size_list, alpha_list


def load_competitor_results(exp_folder, approximation_dict):
    collector = []
    approx_collector = []

    for ts_data_name in approximation_dict.keys():
        dataset_name = ts_data_name_to_dataset_name(ts_data_name)
        approx = approximation_dict[ts_data_name]
        for app in approx:
            competitor_result_file = os.path.join(exp_folder, dataset_name, app, f'competitor_result_{ts_data_name}.csv')
            if not os.path.exists(competitor_result_file):
                continue
            dataset_result = pd.read_csv(competitor_result_file)
            collector.append(dataset_result)
            approx_collector += [app] * dataset_result.shape[0]
    approx_series = pd.Series(approx_collector, name='approximation')
    competitor_results = pd.concat(collector, axis=0).reset_index(drop=True) if len(collector) != 0 else None
    competitor_results_approx = pd.concat((approx_series, competitor_results), axis=1) if len(collector) != 0 else None
    return competitor_results_approx


def load_ground_truth(ts_data_names, data_folder):
    ground_truth = dict()
    ts_length = dict()

    for ts_data_name in ts_data_names:
        if ts_data_name.startswith('Synthetic'):
            number_of_concepts = ts_data_name.split('Concepts_')[0].split('Synthetic_')[1]
            concept_size = ts_data_name.split('dataPerConcept_')[0].split('_')[-1]

            label_file = f'Synthetic_{number_of_concepts}Concepts_{concept_size}dataPerConcept_label.txt'
        else:
            label_file = f'{ts_data_name}_label.txt'

        ground_truth[ts_data_name] = np.loadtxt(f'{data_folder}/{label_file}', delimiter=", ")
        ts_length[ts_data_name] = ground_truth[ts_data_name].size
    return ground_truth, ts_length


def ts_data_name_to_dataset_name(ts_data_name):
    if ts_data_name.startswith('Synthetic_'):
        dataset_name = ts_data_name.split('Concepts_')[-1].split('_')[0]
    else:
        dataset_name = '_'.join(ts_data_name.split('_')[1:])
    return dataset_name


def load_detection_results(exp_folder, approximation_dict, window_length, ol):
    detection_results_in_ts = dict()
    detection_results_in_slidshap = dict()
    for ts_data_name in approximation_dict.keys():
        dataset_name = ts_data_name_to_dataset_name(ts_data_name)
        approximations = approximation_dict[ts_data_name]
        for approximation in approximations:
            slidshap_file = f'{exp_folder}/{dataset_name}/{approximation}/slidshap_drifts_bi_predictions_{ts_data_name}_windowlength{window_length}_overlap{ol}.pkl'
            ts_file = f'{exp_folder}/{dataset_name}/{approximation}/ts_drifts_bi_predictions_{ts_data_name}_windowlength{window_length}_overlap{ol}.pkl'

            if not os.path.exists(ts_file):
                continue
            with open(slidshap_file, 'rb') as f:
                predictions_slidshap = pickle.load(f)
            with open(ts_file, 'rb') as f:
                predictions_ts = pickle.load(f)

            detection_results_in_ts[(ts_data_name, approximation)] = predictions_ts
            detection_results_in_slidshap[(ts_data_name, approximation)] = predictions_slidshap

    return detection_results_in_ts, detection_results_in_slidshap
