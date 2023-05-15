import __shapley_calculation as sv

from tqdm import tqdm
import numpy as np

import time
def compute_shapleyvalues(_mydata, _type, _subsets_bound=-1, approx='max'):
    
    if _type == 'bounded':
        SVC = sv.ShapleyValueCalculator("total_correlation", "subsets_bounded", _subsets_bound)
    elif _type == 'full':
        SVC = sv.ShapleyValueCalculator("total_correlation", "subsets")
    shapleys = SVC.calculate_SVs(_mydata)

    return shapleys


def run_slidshaps(_mydata, _window_width, _overlap, _type, _subsets_bound, _approx):
    
    T_in = 0
    T_fin = _window_width

    data_current = _mydata[T_in:T_fin]
    
    data_shaps = np.asarray(compute_shapleyvalues(data_current.T, _type, _subsets_bound, _approx)).reshape(-1,1)

    start_time = time.time()

    for T_in in tqdm(range(_overlap, np.shape(_mydata.T)[1], _overlap)):
        T_fin = T_in + _window_width
        data_current = _mydata[T_in:T_fin]
        shapleyvalues = np.asarray(compute_shapleyvalues(data_current.T, _type, _subsets_bound, _approx)).reshape(-1,1)
        data_shaps = np.concatenate([data_shaps, shapleyvalues], axis=1)
    running_time = time.time() - start_time

    return data_shaps, running_time