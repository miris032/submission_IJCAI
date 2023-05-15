import itertools
import json
import os

import numpy as np
import pandas as pd
from util import load_detection_results, load_competitor_results, load_ground_truth


def _bi_to_pos(bi_list):
    result = []
    for k, v in enumerate(bi_list):
        if v == 1:
            result.append(k)
    return result


def load_param(exp_folder):
    with open(os.path.join(exp_folder, 'configuration', 'exp_data_config.json'), 'rb') as f:
        exp_data_config = json.load(f)
    with open(os.path.join(exp_folder, 'configuration', 'exp_model_config.json'), 'rb') as f:
        exp_model_config = json.load(f)

    window_length_list = list()
    overlap_list = list()
    buffer_size_list = list()
    ts_data_name_set = set()
    dataset_name_set = set()
    approximation_dict = dict()

    for data in exp_data_config.keys():
        window_length_list += exp_data_config[data]['window_length']
        overlap_list += exp_data_config[data]['overlap']
    buffer_size_list += exp_model_config['slidshaps']['buffer_size']

    for root, dirs, files in os.walk(exp_folder):
        if os.path.basename(root) != exp_folder.split('/')[-1] and not os.path.basename(root).startswith('.'):
            dataset_name_set.add(os.path.basename(root))
        approx = root.split('/')[-1]
        for file in files:
            if file.startswith('slidshap_drifts_') and '_windowlength' in file:
                ts_data_name = file.split('predictions_')[-1].split('_windowlength')[0]
                ts_data_name_set.add(ts_data_name)
                approximation_dict[ts_data_name] = [approx] if ts_data_name not in approximation_dict.keys() else \
                approximation_dict[ts_data_name] + [approx]
    return list(ts_data_name_set), list(set(window_length_list)), list(set(overlap_list)), buffer_size_list, list(
        dataset_name_set), approximation_dict


class DriftDetectionEvaluator:
    """This class is used to evaluate a drift detection method."""

    @staticmethod
    def calculate_dl_tp_fp_fn(located_points, actual_points, interval, total_num):
        # adapted from: https://github.com/alipsgh/tornado/blob/master/evaluators/detector_evaluator.py
        actual_drift_points = actual_points.copy()

        num_actual_drifts = len(actual_points)
        num_located_drift_points = len(located_points)

        drift_detection_tp = 0
        drift_detection_dl = []
        delays = []
        for located in located_points:
            for actual in actual_points:
                if actual <= located <= actual + interval:
                    drift_detection_tp += 1
                    delays.append(located - actual)
                    drift_detection_dl.append(located - actual - (actual_drift_points.index(actual) + 1))
                    actual_points.remove(actual)
                    break
        drift_detection_dl = sum(drift_detection_dl) + (num_actual_drifts - len(drift_detection_dl)) * interval
        drift_detection_dl /= num_actual_drifts
        drift_detection_fp = num_located_drift_points - drift_detection_tp
        drift_detection_fn = num_actual_drifts - drift_detection_tp

        delay_mean = np.mean(delays)
        delay_std = np.std(delays)

        accuracy = (total_num - drift_detection_fp - drift_detection_fn) / total_num
        precision = drift_detection_tp / (
                drift_detection_tp + drift_detection_fp) if drift_detection_tp + drift_detection_fp != 0 else np.nan
        recall = drift_detection_tp / (
                drift_detection_tp + drift_detection_fn) if drift_detection_tp + drift_detection_fn != 0 else np.nan
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall != 0 else np.nan

        return drift_detection_dl, drift_detection_tp, drift_detection_fp, drift_detection_fn, accuracy, precision, recall, f1, delay_mean, delay_std

    def labeling(self, ts_len, groundtruth, ts_win_len, ts_overlap, drift_detector_buf_len,
                 drift_detector_buf_overlap=1):
        time_series_window_labels = [0 for _ in range(int(np.floor((ts_len - ts_win_len) / ts_overlap)))]
        slidSHAP_detection_labels = [0 for _ in range(len(time_series_window_labels) - 2 * drift_detector_buf_len + 1)]
        ts_drift_windows = [(idx - 1, idx) for idx, value in enumerate(groundtruth) if value == 1]
        for start_pos, end_pos in ts_drift_windows:  # label only the middle window at the drift position as drift
            start_win = self.index_mapping_ts_to_win(start_pos, ts_win_len, ts_overlap)[0]
            end_win = self.index_mapping_ts_to_win(end_pos, ts_win_len, ts_overlap)[1]
            time_series_window_labels[int(np.floor((start_win + end_win) / 2))] = 1

            start_shap_idx, end_shap_idx = self.index_mapping_win_to_shap(int(np.floor((start_win + end_win) / 2)),
                                                                          drift_detector_buf_len,
                                                                          drift_detector_buf_overlap)
            slidSHAP_detection_labels[int(np.floor((start_shap_idx + end_shap_idx) / 2))] = 1
        return time_series_window_labels, slidSHAP_detection_labels

    @staticmethod
    def index_mapping_ts_to_win(ts_index, ts_win_len, ts_overlap):
        # all index start counting from 0
        start_win_idx = 0 if ts_index < ts_win_len else np.ceil((ts_index - ts_win_len) / ts_overlap)
        end_win_idx = np.floor(ts_index / ts_overlap)

        return int(start_win_idx), int(end_win_idx)

    @staticmethod
    def index_mapping_win_to_shap(win_index, drift_detector_buf_len, drift_detector_buf_overlap):
        # all index start counting from 0. We maintain two sliding detector buffers with the same size, which act as
        # the historical and latest data buffer. We label the position where the front buffer's end position reaches
        # as the detected drift position. So all 'drift_detector_buf_len' are doubled in this function
        start_shap_idx = max(0, np.ceil(
            (win_index - drift_detector_buf_len * 2) / drift_detector_buf_overlap))
        end_shap_idx = np.floor(win_index / drift_detector_buf_overlap)

        return start_shap_idx, end_shap_idx

    def run_eval(self, exp_folder, config_folder, data_folder):
        eval_result = []
        ts_data_name_list, window_length_list, overlap_list, buffer_size_list, dataset_name_list, approximation_dict \
            = load_param(exp_folder)
        competitor_results = load_competitor_results(exp_folder, approximation_dict)

        model_config = json.load(open(os.path.join(config_folder, 'model_config.json')))

        ground_truth, ts_length = load_ground_truth(ts_data_name_list, data_folder)
        for (ts_data_name, window_length) in itertools.product(ts_data_name_list, window_length_list):
            for approximation in approximation_dict[ts_data_name]:
                print(f'Running evaluation on: {ts_data_name}')
                if competitor_results is not None:
                    # collect competitor results
                    subset = competitor_results[(competitor_results.approximation == approximation) &
                                                (competitor_results.dataset == ts_data_name) & (
                                                            competitor_results.win_len == window_length)]
                    print(subset.shape)
                    if subset.shape[0] != 0:
                        for model in ['hdddm', 'kdqtree', 'adwin', 'pcacd']:
                            delta = window_length * model_config[model]['delta_factor']
                            if model in subset.columns:
                                drift_detection_dl, drift_detection_tp, drift_detection_fp, drift_detection_fn, acc, precision, recall, f1, delay_mean, delay_std = self.calculate_dl_tp_fp_fn(
                                    _bi_to_pos(subset[model]), _bi_to_pos(subset['label']), delta, subset.shape[0])
                                eval_result.append(
                                    [model, ts_data_name, approximation, np.nan, window_length, np.nan, np.nan, np.nan, np.nan,
                                     drift_detection_tp,
                                     drift_detection_fp,
                                     drift_detection_fn, acc, precision, recall, f1, delay_mean, delay_std])
                # collect slidshaps results
                model = 'slidshaps'
                delta = window_length * model_config[model]['delta_factor']
                gamma_list = model_config[model]['gamma']
                alpha_list = model_config[model]['alpha']
                statistical_test_list = model_config[model]['statistical_test']
                for ol in overlap_list:
                    for alpha in alpha_list:
                        for rf in gamma_list:
                            for dbs in buffer_size_list:
                                for statistical_test in statistical_test_list:
                                    overlap = int(window_length * ol)
                                    detection_results_in_ts, detection_results_in_slidshap = load_detection_results(
                                        exp_folder,
                                        approximation_dict, window_length, overlap)
                                    if (ts_data_name, approximation) not in detection_results_in_ts.keys():
                                        print(ts_data_name, approximation, '*'*20)
                                        continue
                                    if not (window_length, overlap, dbs, alpha, rf, statistical_test) in \
                                           detection_results_in_ts[(ts_data_name, approximation)].keys():
                                        continue
                                    detected_positions_in_ts = detection_results_in_ts[(ts_data_name, approximation)][
                                        (window_length, overlap, dbs, alpha, rf, statistical_test)]
                                    drift_detection_dl, drift_detection_tp, drift_detection_fp, drift_detection_fn, acc, precision, recall, f1, delay_mean, delay_std = self.calculate_dl_tp_fp_fn(
                                        _bi_to_pos(detected_positions_in_ts), _bi_to_pos(ground_truth[ts_data_name]),
                                        delta,
                                        ground_truth[ts_data_name].size)
                                    eval_result.append(
                                        [model, ts_data_name, approximation, alpha, window_length, overlap, rf, dbs,
                                         statistical_test,
                                         drift_detection_tp, drift_detection_fp, drift_detection_fn, acc, precision,
                                         recall,
                                         f1, delay_mean, delay_std])
        eval_df = pd.DataFrame(eval_result,
                               columns=['model', 'dataset', 'approximation', 'alpha', 'window_width', 'overlap', 'rf',
                                        'dbs',
                                        'statistical_test', 'tp', 'fp',
                                        'fn', 'acc', 'precision', 'recall', 'f1', 'delay_mean', 'delay_std'])

        eval_df.to_csv(os.path.join(exp_folder, f'eval_result.csv'), index=False)
        return eval_df
