import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from movie_decoding.dataloader.patients import Patient, Patients


def load_patients(patients_id: Union[str, List[str]], file_path: Path) -> Patients:
    patients = Patients()
    for patient_id in patients_id:
        patient_file = file_path / f"patients_{patient_id}.json"
        patients.read_json(patient_file, patient_id)

    return patients
