import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

from brain_decoding.dataloader.patients import Patient, Patients


def load_patients(patients_id: Union[str, List[str]], file_path: Path) -> Patients:
    patients = Patients()

    if isinstance(patients_id, str):
        patients_id = [patients_id]

    for patient_id in patients_id:
        patient_file = file_path / f"patient_{patient_id}.json"
        patients.read_json(patient_file, patient_id)

    return patients
