import numpy as np
import pandas as pd
import torch
from torch.utils.data import dataset


class Meter:
    def __init__(self, fold):
        self.loss = []
        self.f1 = []
        self.accuracy = []
        self.fold = fold

    def add(self, loss, f1, accuracy):
        self.loss.append(loss)
        self.f1.append(f1)
        self.accuracy.append(accuracy)

    def dump(self):
        items = [self.loss, self.f1, self.accuracy]
        res = []
        for i in items:
            res.append(np.mean(i))
        return np.round(np.array(res), 3)

    def dump_wandb(self):
        items = [self.loss, self.f1, self.accuracy]

        names = [
            "fold {} train loss".format(self.fold + 1),
            "fold {} train f1 score".format(self.fold + 1),
            "fold {} train accuracy".format(self.fold + 1),
        ]
        return {n: np.mean(i) for n, i in zip(names, items)}


class ValidMeter:
    def __init__(self, fold):
        self.loss = []
        self.f1 = []
        self.accuracy = []
        self.class_name = ["No", "Pikachu"]
        self.fold = fold

    def add(self, loss, f1, accuracy):
        self.loss.append(loss)
        self.f1.append(f1)
        self.accuracy.append(accuracy)

    def dump(self):
        items = [self.loss, self.f1, self.accuracy]
        res = []
        for i in items:
            res.append(np.mean(i))
        return np.round(np.array(res), 3)

    def dump_wandb(self):
        items = [self.loss, self.f1, self.accuracy]

        names = [
            "fold {} valid loss".format(self.fold + 1),
            "fold {} valid f1 score".format(self.fold + 1),
            "fold {} valid accuracy".format(self.fold + 1),
        ]
        return {n: np.mean(i) for n, i in zip(names, items)}


class TestMeter:
    def __init__(self, fold):
        self.loss = []
        self.f1 = []
        self.accuracy = []
        self.class_name = ["No", "Pikachu"]
        self.fold = fold

    def add(self, loss, f1, accuracy):
        self.loss.append(loss)
        self.f1.append(f1)
        self.accuracy.append(accuracy)

    def dump(self):
        items = [self.loss, self.f1, self.accuracy]
        res = []
        for i in items:
            res.append(np.mean(i))
        return np.round(np.array(res), 3)

    def dump_wandb(self):
        items = [self.loss, self.f1, self.accuracy]

        names = [
            "fold {} test loss".format(self.fold + 1),
            "fold {} test f1 score".format(self.fold + 1),
            "fold {} test accuracy".format(self.fold + 1),
        ]
        return {n: np.mean(i) for n, i in zip(names, items)}
