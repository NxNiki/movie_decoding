from typing import Dict, Optional

import numpy as np


class Meter:
    def __init__(self, fold):
        self.loss = []
        self.f1 = []
        self.accuracy = []
        self.fold = fold

    def add(self, loss: float, f1: float, accuracy: float):
        self.loss.append(loss)
        self.f1.append(f1)
        self.accuracy.append(accuracy)

    def dump(self) -> Optional[np.ndarray[float]]:
        items = [self.loss, self.f1, self.accuracy]
        res = []
        for item in items:
            res.append(np.mean(item))
        return np.round(np.array(res), 3)

    def dump_wandb(self) -> Dict[str, float]:
        items = [self.loss, self.f1, self.accuracy]

        names = [
            "fold {} train loss".format(self.fold + 1),
            "fold {} train f1 score".format(self.fold + 1),
            "fold {} train accuracy".format(self.fold + 1),
        ]
        return {n: np.mean(i) for n, i in zip(names, items)}


class ValidMeter(Meter):
    def __init__(self, fold):
        super().__init__(fold)
        self.class_name = ["No", "Pikachu"]


class TestMeter(Meter):
    def __init__(self, fold):
        super().__init__(fold)
        self.class_name = ["No", "Pikachu"]
