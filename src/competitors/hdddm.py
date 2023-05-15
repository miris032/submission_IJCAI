from abc import ABC
import numpy as np

from menelaus.data_drift.hdddm import HDDDM
from .competitorModel import Competitor


class HDDDM_Competitor(Competitor, ABC):
    def __init__(self, batch_size, dataset, seed=42):
        super().__init__(dataset)
        self.batch_size = batch_size
        self.seed = seed

    def run(self):
        print(f'HDDDM drift detector: batch_size={self.batch_size}')
        np.random.seed(self.seed)
        hdddm = HDDDM()

        detected_drift_batch = [0]
        label_drift_batch = [0]
        drift_prediction = [0 for _ in range(self.batch_size)]

        reference = self.dataset.iloc[:self.batch_size, :-1]
        all_test = self.dataset.iloc[self.batch_size:, :]

        hdddm.set_reference(reference)
        for i in range((all_test.shape[0]) // self.batch_size):
            subset_data = all_test.iloc[self.batch_size * i:self.batch_size * (i + 1), :]
            hdddm.update(subset_data.iloc[:, :-1])
            detected_drift_batch.append(hdddm.drift_state)
            for j in range(self.batch_size - 1):
                drift_prediction.append(0)
            drift_prediction.append(1 if hdddm.drift_state == 'drift' else 0)
            label_drift_batch.append(min(sum(subset_data.iloc[:, -1]), 1))
        for _ in range((all_test.shape[0]) % self.batch_size):
            drift_prediction.append(0)
        assert len(drift_prediction) == self.dataset.shape[0], f'wrong prediction length: {len(drift_prediction)} != {self.dataset.shape[0]}'
        return drift_prediction
