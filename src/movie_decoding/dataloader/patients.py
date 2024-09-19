import json
import warnings
from copy import deepcopy
from logging import warning
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np
from pydantic import BaseModel, Field


# Define the Event model
class Event(BaseModel):
    """
    a list of int of the timestamps of a single event during experiment.
    """

    values: List[int] = Field(..., description="timestamp index of event during experiment")
    description: Optional[str] = None

    def extend_event(self, other: "Event") -> None:
        if not isinstance(other, Event):
            raise ValueError("operands of + must be Events!")

        self.values = self.values + other.values
        if self.description != other.description:
            if self.description and other.description:
                self.description = self.description + " + " + other.description
            else:
                self.description = self.description or other.description

    def add_offset(self, val: int) -> None:
        """
        add val to all elements in event values. This is used to adjust the timestamp of events.
        :param val:
        :return:
        """
        self.values = [value + val for value in self.values]


class Events(BaseModel):
    """
    A dict with event object as values and str as labels.

    set events:
    this will overwrite existing event object in events:
    events['ev1'] = event
    this will raise error if event label exists:
    events.add_event(label, val, des)
    """

    events: Dict[str, Event] = Field(default_factory=dict, description="Dictionary of shared events")

    def __getitem__(self, item: str) -> Event:
        return self.events.get(item, Event(values=[], description=None))

    def __setitem__(self, item: str, event: Event):
        if item in self.events:
            warnings.warn(f"overwrite existing item: {item}")
        self.events[item] = event

    def extend_events(self, other: "Events") -> None:
        events_name = set(self.events.keys() | other.events.keys())
        events_name_common = set(self.events.keys() & other.events.keys())
        for event_name in events_name:
            if event_name in events_name_common:
                self.events[event_name].extend_event(other.events[event_name])
            elif event_name in other.events.keys():
                self.events[event_name] = other.events[event_name]

    def add_event(self, label: str, values: Optional[List[int]] = None, description: Optional[str] = None):
        if self.has_event(label):
            raise KeyError(f"Event {label} already exists in the shared events!")
        self.events[label] = Event(values=values, description=description)

    def get_event(self, label: str) -> Event:
        if not self.has_event(label):
            warnings.warn(f"Event {label} does not exist in the shared events!")
        return self[label]

    def has_event(self, event_name: str) -> bool:
        return event_name in self.events

    @property
    def events_name(self) -> List[str]:
        return list(self.events.keys())

    def add_offset(self, val: int) -> None:
        for event_name in self.events:
            self.events[event_name].add_offset(val)


# Define the Experiment model
class Experiment(BaseModel):
    events: Events = Field(default_factory=Events, description="Events within the experiment")
    neural_data_file: Optional[str] = None
    _neural_data: Optional[np.ndarray] = None

    def __getitem__(self, event_name: str) -> Event:
        return self.events.get_event(event_name)

    def __setitem__(self, event_name: str, values: List[int]):
        self.events.add_event(event_name, values=values)

    @property
    def neural_data(self):
        if self._neural_data is None:
            self._neural_data = self.load_neural_data()
        return self._neural_data

    def load_neural_data(self) -> Optional[np.ndarray]:
        # load neural data (e.g., from a file or database)
        print(f"load neural data: {self.neural_data_file}...")
        pass

    def add_events(self, events: Events):
        """
        add additional events to experiment. will raise error if label exists.
        :param events:
        :return:
        """
        for label in events.events:
            self.events.add_event(
                label=label, values=events.events[label].values, description=events.events[label].description
            )

    def extend_events(self, events: Union[Events, "Experiment"], offset: int):
        """
        add new event values to the end of existing events.
        :param events:
        :param offset:
        :return:
        """

        if not isinstance(events, Events):
            events = events.events

        events = deepcopy(events)
        events.add_offset(offset)
        self.events.extend_events(events)

    def add_offset(self, val: int) -> None:
        self.events.add_offset(val)
        # TO DO: add offset to neural data.

    @property
    def events_name(self):
        return self.events.events_name


# Define the Patient model
class Patient(BaseModel):
    experiments: Dict[str, Experiment] = Field(
        default_factory=dict, description="Dictionary of experiments for the patient"
    )

    def __getitem__(self, experiment_name: str) -> Optional[Experiment]:
        if not self.has_experiment(experiment_name):
            raise KeyError(f"experiment: {experiment_name} not exist!")
        return self.experiments[experiment_name]

    def __setitem__(self, experiment_name: str, experiment: Experiment):
        self.experiments[experiment_name] = experiment

    @property
    def experiments_name(self) -> List[str]:
        return list(self.experiments.keys())

    def add_experiment(self, experiment_name: str):
        if experiment_name in self.experiments:
            warnings.warn(f"experiment: {experiment_name} already exists!")
        else:
            self.experiments[experiment_name] = Experiment()

    def add_event(self, experiment_name: str, event_name: str, values: List[int]):
        if experiment_name not in self.experiments:
            raise KeyError(f"experiment: {experiment_name} not exist!")
        self.experiments[experiment_name].events.add_event(event_name, values)

    def add_events(self, experiment_name: str, events: Events):
        for event_name, event in events.events.items():
            values = event.values
            self.add_event(experiment_name, event_name, values)

    @property
    def events_name(self) -> List[str]:
        """
        Retrieve a list of all unique event keys across all experiments in this patient.
        """
        event_keys_set: Set[str] = set()

        for experiment in self.experiments.values():
            event_keys_set.update(experiment.events.events_name)

        return list(event_keys_set)

    def has_experiment(self, experiment_name: str) -> bool:
        return experiment_name in self.experiments

    def has_event(self, experiment_name: str, event_name: str) -> bool:
        return self.has_experiment(experiment_name) and self.experiments[experiment_name].has_event(event_name)

    def export_json(self, file_name: Union[Path, str]) -> None:
        if isinstance(file_name, str):
            file_name = Path(file_name)

        if not file_name.parent.exists():
            file_name.parent.mkdir(parents=True, exist_ok=True)

        with open(file_name, "w") as json_file:
            json_file.write(self.model_dump_json(indent=4))
        print(f"Model exported to {file_name}")


# Define the overall Patients model
class Patients(BaseModel):
    patients: Dict[str, Patient] = Field(default_factory=dict, description="Dictionary of patients")

    def __getitem__(self, patient_id: str) -> Optional[Patient]:
        if patient_id not in self.patients:
            raise KeyError(f"Patient: {patient_id} does not exist!")
        return self.patients[patient_id]

    def __setitem__(self, patient_id: str):
        self.add_patient(patient_id)

    @property
    def patients_id(self) -> List[str]:
        return list(self.patients.keys())

    @property
    def experiments_name(self) -> List[str]:
        experiment_keys_set: Set[str] = set()
        for patient in self.patients.values():
            experiment_keys_set.update(patient.experiments_name)
        return list(experiment_keys_set)

    @property
    def events_name(self) -> List[str]:
        """
        Retrieve a list of all unique event keys across all experiments and all patients.
        """
        event_keys_set: Set[str] = set()
        for patient in self.patients.values():
            event_keys_set.update(patient.events_name)
        return list(event_keys_set)

    def add_patient(self, patient_id: str):
        if patient_id not in self.patients:
            self.patients[patient_id] = Patient()
        else:
            warning(f"Patient: {patient_id} already exists!")

    def add_experiment(self, patient_id: str, experiment_name: str):
        print(f"patients.add_experiment: {experiment_name} to patient: {patient_id}")
        if not self.has_patient(patient_id):
            self.add_patient(patient_id)
        if experiment_name in self.patients[patient_id]:
            warnings.warn(f"experiment: {experiment_name} exists in patient: {patient_id}")
        else:
            self.patients[patient_id].add_experiment(experiment_name)

    def add_event(self, patient_id: str, experiment_name: str, event_name: str, values: List[int]):
        self.patients[patient_id][experiment_name].events.add_event(event_name, values)

    def add_events(self, patient_id: Union[str, List[str]], experiment_name: Union[str, List[str]], events: Events):
        if patient_id is None:
            patient_id = self.patients_id

        if experiment_name is None:
            experiment_name = self.experiments_name

        for patient_id in patient_id:
            for experiment_name in experiment_name:
                self.patients[patient_id][experiment_name].events.extend_events(events)

    def has_patient(self, patient_id: str) -> bool:
        return patient_id in self.patients

    def has_experiment(self, patient_id: str, experiment_name: str) -> bool:
        return self.has_patient(patient_id) and self.patients[patient_id].has_experiment(experiment_name)

    def has_event(self, patient_id: str, experiment_name: str, event_name: str) -> bool:
        return (
            self.has_patient(patient_id)
            and self.patients[patient_id].has_experiment(experiment_name)
            and self.patients[patient_id][experiment_name].has_event(event_name)
        )

    def export_json(self, file_path: Union[Path, str]) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        for patient_id in self.patients_id:
            file_name = file_path / f"patient_{patient_id}.json"
            self.patients[patient_id].export_json(file_name)

    def read_json(self, file_name: Union[Path, str], patient_id: Union[str, int]) -> None:
        if not isinstance(patient_id, str):
            patient_id = str(patient_id)

        print(f"Patients.read_json: patient_id: {patient_id}")
        print(f"Patients.read_json: read patient file: \n{file_name}")
        self.patients[patient_id] = Patient.model_validate_json(open(file_name).read())


# Example usage within the module (can be removed or commented out for production use)
if __name__ == "__main__":
    patients_data = Patients()

    # Using direct assignment for events
    patients_data.add_experiment(patient_id="567", experiment_name="free_recall1")
    patients_data["567"]["free_recall1"]["LA"] = [1234, 23456]
    patients_data["567"]["free_recall1"]["CIA"] = [12343, 234256]
    patients_data["567"]["free_recall1"]["white house"] = [7890, 6789]

    patients_data.add_experiment(patient_id="890", experiment_name="cued_recall1")
    patients_data["890"]["cued_recall1"]["CIA/FBI"] = [11223, 44556]
    patients_data["890"]["cued_recall1"]["CIA/FBI"].description = "details of event"

    print(patients_data["567"]["free_recall1"]["LA"])
    print(patients_data["567"]["free_recall1"].neural_data)
    print(patients_data["567"]["free_recall1"].events_name)
    print(patients_data["567"].events_name)
    print(patients_data.events_name)
    print(patients_data["890"]["cued_recall1"]["CIA/FBI"].description)

    print("----test non-exist event:----")
    print(patients_data["567"]["free_recall1"]["non-exist"])
    print(patients_data["567"]["free_recall1"]["non-exist"].values)

    events1 = Events()
    events1.add_event(label="LA", values=[1, 2, 3], description="LA1")
    events2 = Events()
    events2.add_event(label="LA", values=[1, 2, 3])
    events2.add_event(label="CIA", values=[1, 2, 3], description="CIA")
    events2.add_offset(10)
    events1.extend_events(events2)

    print("----test extend_events:----")
    print(events1["LA"])

    patient = Patient(patient_id="111")
    patient.add_experiment("exp1")
    patient.add_events("exp1", events1)
    events2 = Events()
    events2.add_event(label="ev3", values=[1, 2, 3], description="ev3")
    patient["exp1"].add_events(events2)
