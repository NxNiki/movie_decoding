import numpy as np
from torchvision.transforms import transforms


class RandomLength():
    def __init__(self, movie_sampling_rate, min_keep_ratio=0.8):
        self.movie_sampling_rate = movie_sampling_rate
        self.min_keep_ratio = min_keep_ratio

    def __call__(self, input_tensor):
        # length = int(np.rint(self.movie_sampling_rate * np.random.uniform(1, 2)))
        min_length = self.min_keep_ratio * self.movie_sampling_rate
        max_length = self.movie_sampling_rate
        start_index = np.random.randint(0, max_length - min_length)
        slice_length = np.random.randint(min_length, max_length - start_index)
        new_input = np.zeros_like(input_tensor)
        new_input[:, start_index:start_index + slice_length] = input_tensor[:, start_index:start_index + slice_length]
        return new_input
        