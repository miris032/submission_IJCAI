from abc import ABC
import numpy as np

from menelaus.data_drift.pca_cd import PCACD
from .competitorModel import Competitor


class PCACD_Competitor(Competitor, ABC):
    def __init__(self, dataset, window_length):
        super().__init__(dataset)
        self.window_length = window_length

    def run(self):
        print(f'pca-cd drift detector: window={self.window_length}')
        pca_cd = PCACD(window_size=self.window_length, divergence_metric="intersection")
        try:
            drift_prediction = []
            test = self.dataset.iloc[:, :-1]
            for i in range(test.shape[0]):
                pca_cd.update(test.iloc[[i]])
                drift_prediction.append(1 if pca_cd.drift_state == 'drift' else 0)

            assert len(drift_prediction) == self.dataset.shape[0], 'wrong prediction length'
        except:
            print(f'Cannot run PCACD on the current data with window length {self.window_length}')
        return drift_prediction
