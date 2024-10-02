from dataclasses import dataclass

from .core import Dictable, StringIntEnum


class RecordingTech(StringIntEnum):
    UTAH_ARRAY_SPIKES = 0
    UTAH_ARRAY_THRESHOLD_CROSSINGS = 1
