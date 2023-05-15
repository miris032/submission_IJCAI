import warnings

import numpy as np
import pandas as pd
import os
import pickle
from pathlib import Path
from scipy import stats

import click


def _slidSHAPPos_to_tsPos(slidSHAPPred, window_width, overlap):
    """
    Given a series of detected drift positions on the slidSHAP series, map them back to the positions in the original
    time series, based on the last timestamp in the sliding window corresponding to each slidSHAP value.
    slidSHAPPred is the binary-encoded slidSHAP predictions. 0 no drift, 1 drift.
    """
    ts_pred = [0 for _ in range(window_width + overlap * (len(slidSHAPPred) - 1))]
    ts_drift_pos = []
    for idx, slidSHAP in enumerate(slidSHAPPred):
        if slidSHAP == 1:
            ts_drift_pos.append(window_width + idx * overlap - 1)
    for pos in ts_drift_pos:
        ts_pred[pos] = 1
    return ts_pred


def run_drift_detection(exp_folder, dataset_name, detection_buf_size, window_length, overlap, alpha, gamma,
                        statistical_test):
    result_buffer = []
    slidshap_bi_prediction_buffer = dict()
    ts_bi_prediction_buffer = dict()
    ol = int(overlap * window_length)
    print(
        f'Drift detection: {dataset_name}, window length{window_length}, overlap {ol}, alpha {alpha}, statistical test {statistical_test}...')
    file_name = f'slidshaps_{dataset_name}_windowlength{window_length}_overlap{ol}'
    file_path = os.path.join(exp_folder, file_name + '.txt')
    output_path = f'{exp_folder}/drifts_{dataset_name}_windowlength{window_length}_overlap{ol}_buf_{detection_buf_size}.csv'
    assert os.path.exists(file_path), f'Cannot fine {file_path}'

    if os.path.exists(output_path):
        warnings.warn('drift detection result already exists: ' + output_path)
        return

    shaps = np.loadtxt(file_path, delimiter=",").T
    detected_drifts, slidshap_bi_pred, p_values = _detect(shaps, detection_buf_size, alpha, gamma, statistical_test) # todo:remvoe
    ts_bi_pred = _slidSHAPPos_to_tsPos(slidshap_bi_pred, window_length, ol)
    slidshap_bi_prediction_buffer[
        (window_length, ol, detection_buf_size, alpha, gamma, statistical_test)] = slidshap_bi_pred
    ts_bi_prediction_buffer[(window_length, ol, detection_buf_size, alpha, gamma, statistical_test)] = ts_bi_pred

    for drifts in detected_drifts:
        result_buffer.append(
            [dataset_name, alpha, window_length, ol, detection_buf_size, drifts[0], drifts[1], statistical_test])
    df = pd.DataFrame(result_buffer,
                      columns=['dataset', 'alpha', 'window_width', 'overlaps', 'detection_buf_size',
                               'slidshap_drift_pos',
                               'p_value', 'statistical_test'])
    with open(f'{exp_folder}/slidshap_drifts_bi_predictions_{dataset_name}_windowlength{window_length}_overlap{ol}.pkl', 'wb') as f:
        pickle.dump(slidshap_bi_prediction_buffer, f)
    with open(f'{exp_folder}/ts_drifts_bi_predictions_{dataset_name}_windowlength{window_length}_overlap{ol}.pkl', 'wb') as f:
        pickle.dump(ts_bi_prediction_buffer, f)
    df.to_csv(output_path, index=False)
    p_values.to_csv(f'{exp_folder}/drifts_{dataset_name}_windowlength{window_length}_overlap{ol}_pvalues.csv', header=False, index=False) # todo:remove


@click.command()
@click.option('--dataset_name', type=str)
@click.option('--detection_buf_size', type=int)
@click.option('--arg_win_width', type=str)
@click.option('--arg_overlap', type=str)
@click.option('--arg_alphas', type=str, default='0.01,0.05')
def main(dataset_name, detection_buf_size, arg_win_width, arg_overlap, arg_alphas):
    ww = int(arg_win_width)
    ol = int(arg_overlap)
    alphas = [float(x) for x in arg_alphas.split(',')]
    result_buffer = []
    prediction_buffer = dict()
    for alpha in alphas:
        print(f'Drift detection: {dataset_name}, window {ww}, overlap {ol}, alpha {alpha}...')
        file_name = f'shapleyvalues_data{dataset_name}_windowwidth{ww}_overlap{ol}'
        root_path = os.path.dirname(Path(os.path.realpath(__file__)).parent)
        file_path = f'{root_path}/results/{file_name}.txt'
        if not os.path.exists(file_path):
            print(f'Cannot fine {file_path}')
            continue
        shaps = np.loadtxt(file_path, delimiter=",").T

        detected_drifts, predictions = _detect(shaps, detection_buf_size, alpha)
        prediction_buffer[(ww, ol, alpha)] = predictions
        for drifts in detected_drifts:
            result_buffer.append([dataset_name, alpha, ww, ol, detection_buf_size, drifts[0], drifts[1]])
    df = pd.DataFrame(result_buffer,
                      columns=['dataset', 'alpha', 'window_width', 'overlaps', 'detection_buf_size', 'drift_pos',
                               'p_value'])
    with open(f'{root_path}/results/drifts_predictions_{dataset_name}.pkl', 'wb') as f:
        pickle.dump(prediction_buffer, f)
    df.to_csv(f'{root_path}/results/drifts_{dataset_name}_buf_{detection_buf_size}.csv', index=False)


def _detect(shaps, detection_buf_size, alpha, gamma, statistical_test):
    length = shaps.shape[0]
    detected_drifts = []
    p_value_buf = []
    predictions = [0 for _ in range(length)]
    slid = 0
    while slid < length - 2 * detection_buf_size:
        tmp = []
        for i in range(shaps.shape[1]):
            hist = shaps[slid:slid + detection_buf_size, i]
            new = shaps[slid + detection_buf_size:slid + 2 * detection_buf_size, i]
            st, p_value = stats.ks_2samp(hist, new) if statistical_test == 'ks-test' else stats.ttest_ind(hist, new)
            tmp.append(p_value)
        slid += 1
        p_value_buf.append(tmp)
        range_size = int(gamma * detection_buf_size)
        alarm = _check_drift(pd.DataFrame(p_value_buf), range_size, alpha)
        if alarm:
            predictions[slid + 2 * detection_buf_size - 1] = 1
            detected_drifts.append([slid - range_size, min(tmp)])

    predictions = _reduce_ajcent_alarms(predictions)
    return detected_drifts, predictions, pd.DataFrame(p_value_buf) #todo:remove


def _check_drift(p_value_df, k, alpha):
    """input the p values of all dimensions, check whether there are continuously k p_values less than alpha on any
    of the dimensions, if so return True to trigger alarm."""

    if p_value_df.shape[0] < k:
        return False

    for i in range(p_value_df.shape[1]):
        tmp = p_value_df.iloc[-k:, i]
        if tmp[tmp < alpha].size != k:
            continue
        else:
            return True
    return False


def _reduce_ajcent_alarms(predictions, align='left'):
    """
    reduce a series of adjacent drift alarms into one, based on the 'align' argument.
    """
    buffer = []
    tmp = []
    for i in predictions:
        if i == 0 and len(tmp) == 0:
            buffer.append(0)
        elif i == 1:
            tmp.append(1)
        else:
            sub = [0 for _ in range(len(tmp))]
            position = 0 if align == 'left' else len(sub) // 2 if align == 'middle' else -1
            sub[position] = 1
            for x in sub:
                buffer.append(x)
            buffer.append(0)
            tmp = []
    if len(tmp) != 0:
        sub = [0 for _ in range(len(tmp))]
        position = 0 if align == 'left' else len(sub) // 2 if align == 'middle' else -1
        sub[position] = 1
        for x in sub:
            buffer.append(x)
    len_buffer = len(buffer)
    len_pred = len(predictions)
    assert len_buffer == len_pred, f'Different length after reduction: {buffer} != {predictions}'
    return buffer


if __name__ == '__main__':
    main()
