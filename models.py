from typing import List, Optional, Dict
from openenv.core.env_server import Action, Observation, State

class TriageAction(Action):
    triage_level: int                    # 1–5 ESI scale
    care_pathway: str                    # "resuscitation" | "acute" |
                                         # "fast_track" | "observation" |
                                         # "discharge_likely"
    confidence: float                    # 0.0–1.0 agent self-reported confidence

class PatientObservation(Observation):
    patient_id: str
    age: int
    chief_complaint: str
    vitals: Dict[str, float]                         # heart_rate, bp_systolic, bp_diastolic,
                                         #  respiratory_rate, spo2, temperature,
                                         #  gcs_score
    symptoms: List[str]
    medical_history: List[str]
    arrival_mode: str                    # "ambulance" | "walk-in" | "police"
    time_in_waiting_room_minutes: int
    queue_length: int
    task_id: str
    done: bool
    reward: float
    masked: bool                         # True for hard task — masks some vitals
    attempts_remaining: int
    message: str                         # Feedback message to agent

class ERTriageState(State):
    episode_id: str
    step_count: int
    current_patient_id: str
    task_id: str
    target_esi: Optional[int] = None       # Hidden ground truth (for state only)
    max_attempts: int = 10
