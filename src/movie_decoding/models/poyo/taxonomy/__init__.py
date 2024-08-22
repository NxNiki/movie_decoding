from .core import Dictable, StringIntEnum
from .descriptors import (
    DandisetDescription,
    SessionDescription,
    SortsetDescription,
    SubjectDescription,
    to_serializable,
)
from .macaque import Macaque
from .recording_tech import RecordingTech
from .subject import Sex, Species
from .task import Task


class OutputType(StringIntEnum):
    CONTINUOUS = 0
    BINARY = 1
    MULTILABEL = 2
    MULTINOMIAL = 3
