import itertools

from data_generator.synthetic_generator import run_generation
import os
import numpy as np
from pathlib import Path
import shutil
from _slidSHAPs import run_slidshaps
import _loadingdata
from drift_detection import run_drift_detection
from run_competitors import run_competitors
from evaluator import DriftDetectionEvaluator
from util import load_data_parameters, load_slidshaps_parameters


experiment_id = 'exp_2023_ijcai'
datasets = ['Synthetic',  'Cancer', 'LED_10000_10', 'Poker',  'MSL', 'KDD_all', 'LED_10000_10']

competitors = ['hdddm', 'adwin', 'kdqtree', 'pcacd']
root_path = os.path.dirname(Path(os.path.realpath(__file__)).parent)
data_folder = os.path.join(root_path, 'data')
config_folder = os.path.join(root_path, 'configuration')
results_folder = os.path.join(root_path, 'results', experiment_id)


def _run_synthetic_generator(input_data_folder, concept_size, categories_per_feature, nums_concepts, correlation_func):
    run_generation(input_data_folder,
                   concept_size,
                   categories_per_feature,
                   nums_concepts,
                   correlation_func)


def _calc_slidshaps(exp_folder, ts_data_name, window_length, overlap, approximation_type, subsets_bound):
    ol = int(overlap * window_length)
    slidshaps_output_path = os.path.join(exp_folder, f'slidshaps_{ts_data_name}_windowlength{window_length}_overlap{ol}.txt')
    pathruntimes = os.path.join(exp_folder, f'time_{ts_data_name}_windowlength{window_length}_overlap{ol}.txt')

    if os.path.exists(slidshaps_output_path):
        return

    if ts_data_name == 'concept_drifted_data':
        _mydata = _loadingdata.load_data_CD()
    else:
        _mydata = _loadingdata.load_data(ts_data_name)
    print(ts_data_name)
    data_shaps, running_time = run_slidshaps(_mydata, window_length, ol, approximation_type, subsets_bound, _approx='max')


    np.savetxt(slidshaps_output_path, data_shaps, delimiter=", ")
    np.savetxt(pathruntimes, [running_time], delimiter=", ")


def _run_drift_detection(exp_folder, ts_data_name, detection_buf_size, window_length, overlap, alpha, gamma,
                         statistical_test):
    run_drift_detection(exp_folder, ts_data_name, detection_buf_size, window_length, overlap, alpha, gamma,
                        statistical_test)


def _run_competitors(ts_data_name, competitors, label_file, window_length, exp_folder, data_folder):
    run_competitors(ts_data_name, competitors, label_file, window_length, exp_folder, data_folder)


def run_experiments(dataset, use_existing_synthetic_data=True):
    window_length_list, overlap_list, concept_size_list, number_of_concepts_list, categories_per_feature_list, \
    correlation_func_list, approximation_type, subsets_bound_list = load_data_parameters(dataset, config_folder)
    statistical_test_list, gamma_list, buffer_size_list, alpha_list = load_slidshaps_parameters(config_folder)

    if not os.path.exists(os.path.join(results_folder, 'configuration')):
        os.makedirs(os.path.join(results_folder, 'configuration'))
    shutil.copyfile(os.path.join(config_folder, 'data_config.json'),
                    os.path.join(results_folder, 'configuration', 'exp_data_config.json'))
    shutil.copyfile(os.path.join(config_folder, 'model_config.json'),
                    os.path.join(results_folder, 'configuration', 'exp_model_config.json'))

    for (number_of_concepts, correlation_func, concept_size, categories_per_feature, statistical_test, subsets_bound) in \
            itertools.product(number_of_concepts_list,
                              correlation_func_list,
                              concept_size_list,
                              categories_per_feature_list,
                              statistical_test_list,
                              subsets_bound_list):
        if dataset == 'Synthetic':
            sub_folder_name = correlation_func
            ts_data_name = f'Synthetic_{number_of_concepts}Concepts_{correlation_func}_{concept_size}dataPerConcept_' \
                           f'{categories_per_feature}category'
        else:
            sub_folder_name = dataset
            ts_data_name = f'Realworld_{dataset}'
        exp_folder = os.path.join(results_folder, sub_folder_name, approximation_type+str(subsets_bound))
        label_file = f'Synthetic_{number_of_concepts}Concepts_{concept_size}dataPerConcept_label.txt' \
            if dataset == 'Synthetic' else f'{ts_data_name}_label.txt'

        if not os.path.exists(exp_folder):
            os.makedirs(exp_folder)

        if dataset == 'Synthetic':
            if os.path.exists(os.path.join(data_folder, ts_data_name + '.txt')) and not use_existing_synthetic_data:
                os.remove(os.path.join(data_folder, ts_data_name + '.txt'))
                _run_synthetic_generator(data_folder, concept_size, categories_per_feature, number_of_concepts,
                                     correlation_func)
            elif not os.path.exists(os.path.join(data_folder, ts_data_name + '.txt')):
                _run_synthetic_generator(data_folder, concept_size, categories_per_feature, number_of_concepts,
                                         correlation_func)

        for (window_length, overlap, gamma, alpha) in itertools.product(window_length_list, overlap_list, gamma_list,
                                                                        alpha_list):
            _calc_slidshaps(exp_folder, ts_data_name, window_length, overlap, approximation_type, subsets_bound)
            for detection_buf_size in buffer_size_list:
                _run_drift_detection(exp_folder, ts_data_name, detection_buf_size, window_length, overlap, alpha,
                                     gamma, statistical_test)
            _run_competitors(ts_data_name, competitors, label_file, window_length, exp_folder, data_folder)


def run_evaluation(eval_folder):
    driftDetectionEvaluator = DriftDetectionEvaluator()

    driftDetectionEvaluator.run_eval(eval_folder, config_folder, data_folder)


def main():
    for dataset in datasets:
        run_experiments(dataset)
    run_evaluation(results_folder)


if __name__ == '__main__':
    main()
