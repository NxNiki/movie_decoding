import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import wandb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    multilabel_confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from src.param.param_data import LABELS


class Evaluator:
    def __init__(self, config, fold):
        self.config = config
        if config["use_spontaneous"]:
            self.classes = LABELS.append("Spontaneous")
        else:
            self.classes = LABELS

        self.fold = fold

    def calculate_f1(self, y_true, y_pred):
        f1 = f1_score(y_true, y_pred, zero_division=np.nan, average="macro")
        return f1

    def calculate_accuracy(self, y_true, y_pred):
        # accuracy = (y_pred == y_true).sum() / len(y_true)
        # accuracy = accuracy_score(y_true, y_pred)
        acc = []
        for i in range(len(self.classes)):
            acc.append(accuracy_score(y_true[:, i], y_pred[:, i]))
        accuracy = np.mean(acc)
        return accuracy

    def roc_analysis(self, y_true, y_pred, label=None):
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        auc_score = roc_auc_score(y_true, y_pred)

        # Find the threshold that maximizes Youden's index
        youden_index = tpr - fpr
        best_threshold_idx = np.argmax(youden_index)
        best_threshold = thresholds[best_threshold_idx]

        img_path = os.path.join(
            self.config["valid_save_path"],
            "fold{}_val_roc_curve_{}.png".format(self.fold + 1, self.classes[label]),
        )
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.plot([0, 0, 1], [0, 1, 1], linestyle="--", color="gray", alpha=0.5)
        plt.scatter(
            fpr[best_threshold_idx],
            tpr[best_threshold_idx],
            marker="o",
            color="black",
            label=f"Threshold = {best_threshold:.2f}",
        )
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("fold{} {} ROC Curve".format(self.fold + 1, self.classes[label]))
        plt.legend()
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

    def prediction_visualization(self, y_true, y_pred, index, label=None):
        fig, ax = plt.subplots(figsize=(15, 8))
        ax.set_xticks(np.linspace(0, max(index) + 1, 8))
        ax.set_xticklabels(np.linspace(0, max(index) + 1, 8))
        ax.set_xlabel("Time")
        ax.set_yticks(np.arange(1, self.config["num_labels"] * 2 + 1))

        for line_x in index:
            ax.axvline(x=line_x, color="green", linewidth=0.05)

        column_values = []
        for i in range(self.config["num_labels"]):
            column_values.append(self.classes[i] + ": Ture")
            column_values.append(self.classes[i] + ": Pred")
        ax.set_yticklabels(column_values)

        for i in range(self.config["num_labels"]):
            true = y_true[:, i]
            loc = np.where(true == 1)[0]
            loc = index[loc]
            y = [i * 2 + 1] * len(loc)
            ax.scatter(loc, y, color="black", s=0.05)

            pred = y_pred[:, i]
            loc = np.where(pred == 1)[0]
            loc = index[loc]
            y = [i * 2 + 2] * len(loc)
            ax.scatter(loc, y, color="red", s=0.05)

        img_path = os.path.join(
            self.config["valid_save_path"],
            "fold{}_predictions.png".format(self.fold + 1),
        )
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

    def calculate_confusion(self, y_true, y_pred, label=None):
        img_path = os.path.join(
            self.config["valid_save_path"],
            "fold{}_val_cfmatrix_{}.png".format(self.fold + 1, self.classes[label]),
        )
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

        cmd = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["NO", self.classes[label]]
        )
        cmd.plot()
        cmd.figure_.savefig(img_path)
        # df_cm = pd.DataFrame(cm/np.sum(cm) * 100,
        #                      index=[i for i in ['NO', self.classes[label]]],
        #                      columns=[i for i in ['NO', self.classes[label]]])
        # plt.figure()
        # sn.heatmap(df_cm, annot=True)
        # plt.savefig(img_path)
        plt.cla()
        plt.clf()
        plt.close()
        image = wandb.Image(
            img_path,
            caption="Fold {} {} Validation Confusion Matrix".format(
                self.fold + 1, self.classes[label]
            ),
        )
        return image

    def evaluate_metrics(self, y_true, y_pred, frame_index, label=None):
        confusion_map = self.calculate_confusion(y_true, y_pred, label=label)
        wrong_frames = frame_index[np.where(y_true != y_pred)]
        actual_show_frames = frame_index[np.array(y_true, dtype=bool)]
        predict_show_frames = frame_index[np.array(y_pred, dtype=bool)]

        status = {
            "fold{}_confusion_fig_{}".format(
                self.fold + 1, self.classes[label]
            ): confusion_map
        }
        # np.savetxt("wrong_frames_{}".format(self.classes[label]), wrong_frames, fmt='%d')
        # np.savetxt("frames_{}_actual_show".format(self.classes[label]), actual_show_frames, fmt='%d')
        # np.savetxt("frames_{}_predict_show".format(self.classes[label]), predict_show_frames, fmt='%d')
        return status
