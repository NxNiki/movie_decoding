from typing import Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field


# Define the Event model
class Event(BaseModel):
    values: List[int] = Field(..., description="List of integer values for the event")


# Define the Experiment model
class Experiment(BaseModel):
    events: Dict[str, Event] = Field(default_factory=dict, description="Dictionary of events within the experiment")
    neural_data: Optional[np.ndarray] = Field(None, description="Neural recording data as a NumPy array")

    def __getitem__(self, event_name: str) -> Event:
        if event_name not in self.events:
            self.events[event_name] = Event(values=[])
        return self.events[event_name]

    def __setitem__(self, event_name: str, values: List[int]):
        self.events[event_name] = Event(values=values)


# Define the Patient model
class Patient(BaseModel):
    experiments: Dict[str, Experiment] = Field(
        default_factory=dict, description="Dictionary of experiments for the patient"
    )

    def __getitem__(self, experiment_name: str) -> Experiment:
        if experiment_name not in self.experiments:
            self.experiments[experiment_name] = Experiment()
        return self.experiments[experiment_name]

    def __setitem__(self, experiment_name: str, event_data: Dict[str, List[int]]):
        experiment = self.experiments.get(experiment_name, Experiment())
        for event_name, values in event_data.items():
            experiment[event_name] = values
        self.experiments[experiment_name] = experiment


# Define the overall PatientsData model
class PatientsData(BaseModel):
    patients: Dict[str, Patient] = Field(default_factory=dict, description="Dictionary of patients")

    def __getitem__(self, patient_id: str) -> Patient:
        if patient_id not in self.patients:
            self.patients[patient_id] = Patient()
        return self.patients[patient_id]

    def __setitem__(self, patient_id: str, experiment_data: Dict[str, Dict[str, List[int]]]):
        patient = self.patients.get(patient_id, Patient())
        for experiment_name, event_data in experiment_data.items():
            patient[experiment_name] = event_data
        self.patients[patient_id] = patient


# Example usage within the module (can be removed or commented out for production use)
if __name__ == "__main__":
    patients_data = PatientsData()

    # Using direct assignment for events
    patients_data["567"]["free_recall1"]["LA"] = [1234, 23456]

    # Adding neural data to the experiment
    patients_data["567"]["free_recall1"].neural_data = np.array([1.0, 2.0, 3.0])

    print(patients_data["567"]["free_recall1"]["LA"].values)
    print(patients_data["567"]["free_recall1"].neural_data)
