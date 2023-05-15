from abc import ABC

from menelaus.data_drift.kdq_tree import KdqTreeStreaming
from .competitorModel import Competitor


class KdqTree_Competitor(Competitor, ABC):
    def __init__(self, dataset, window_length, alpha=0.01, bootstrap_samples=1000, count_ubound=100):
        super().__init__(dataset)
        self.window_length = window_length
        self.alpha = alpha
        self.bootstrap_samples = bootstrap_samples
        self.count_ubound = count_ubound

    def run(self):
        print(f'kdq-Tree drift detector: window={self.window_length}')
        det = KdqTreeStreaming(window_size=self.window_length,
                               alpha=self.alpha,
                               bootstrap_samples=self.bootstrap_samples,
                               count_ubound=self.count_ubound)

        drift_prediction = []
        test = self.dataset.iloc[:, :-1]
        for i in range(test.shape[0]):
            det.update(test.iloc[[i]])
            drift_prediction.append(1 if det.drift_state == 'drift' else 0)
        assert len(drift_prediction) == self.dataset.shape[0], 'wrong prediction length'
        return drift_prediction
