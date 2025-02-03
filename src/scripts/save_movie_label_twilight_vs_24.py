"""
save labels (24 vs twilight) to train transformer model
"""


import numpy as np

from brain_decoding.config.file_path import MOVIE_LABEL_TWILIGHT_VS_24
from brain_decoding.param.param_data import PREDICTION_FS

# p 570:
SCREENING_DURATION = 40 * 60
TWILIGHT_DURATION = 45 * 60
MOVIE24_DURATION = 1706310981.43703 - 1706308502.12459

n_samples_twilight = int(TWILIGHT_DURATION * PREDICTION_FS)
n_samples_24 = int(MOVIE24_DURATION * PREDICTION_FS)
movie_label = np.zeros((n_samples_twilight + n_samples_24, 2), np.bool)
movie_label[0:n_samples_twilight, 0] = 1
movie_label[n_samples_twilight:, 1] = 1

np.save(MOVIE_LABEL_TWILIGHT_VS_24, movie_label.transpose())
