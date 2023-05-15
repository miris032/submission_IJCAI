from abc import ABC

from skmultiflow.drift_detection.adwin import ADWIN
from .competitorModel import Competitor


class Adwin_Competitor(Competitor, ABC):
    def __init__(self, dataset):
        super().__init__(dataset)

    def run(self):
        print(f'ADWIN drift detector')

        drift_prediction = []
        test = self.dataset.iloc[:, :-1]
        model_list = []
        for dim in range(test.shape[1]):
            model_list.append(ADWIN())

        for i in range(test.shape[0]):
            drift = False
            for dim in range(test.shape[1]):
                model_list[dim].add_element(test.iloc[i, dim])
                if model_list[dim].detected_change():
                    drift = True
                    for d in range(test.shape[1]):
                        model_list[d].reset()
                    break
            drift_prediction.append(1 if drift else 0)

        assert len(drift_prediction) == self.dataset.shape[0], 'wrong prediction length'
        return drift_prediction
