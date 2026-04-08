import random
import uuid
import json
import os

from openenv.core.env_server import Environment
from models import TriageAction, PatientObservation, ERTriageState

# Load patient cases
CASES_PATH = os.path.join(os.path.dirname(__file__), "cases.json")
with open(CASES_PATH) as f:
    ALL_CASES = json.load(f)


class ERTriageEnvironment(Environment):

    SUPPORTS_CONCURRENT_SESSIONS = True
    MAX_ATTEMPTS = 10

    def __init__(self):
        self._state = None
        self._current_case = None
        self._target_esi = None
        self._target_pathway = None
        self._remaining = self.MAX_ATTEMPTS
        self._done = False

    # -------------------------
    # RESET
    # -------------------------
    def reset(self, seed=None, episode_id=None, **kwargs) -> PatientObservation:
        if seed is not None:
            random.seed(seed)

        task_id = kwargs.get("task_id", "classic_presentations")
        task_cases = [c for c in ALL_CASES if c["task_id"] == task_id]

        case = random.choice(task_cases)
        self._current_case = case

        self._target_esi = case["ground_truth_esi"]
        self._target_pathway = case["ground_truth_pathway"]
        self._remaining = self.MAX_ATTEMPTS
        self._done = False

        self._state = ERTriageState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            current_patient_id=case["patient_id"],
            task_id=task_id,
            target_esi=None,
            max_attempts=self.MAX_ATTEMPTS
        )

        return self._build_observation(
            reward=0.0,
            message="New patient. Assign ESI (1-5) and care pathway."
        )

    # -------------------------
    # STEP
    # -------------------------
    def step(self, action: TriageAction, **kwargs) -> PatientObservation:
        if self._done:
            return self._build_observation(
                reward=0.0,
                message="Episode already finished."
            )

        self._state.step_count += 1
        self._remaining -= 1

        reward, message, correct = self._compute_reward(action)

        if correct or self._remaining <= 0:
            self._done = True

        return self._build_observation(
            reward=reward,
            message=message
        )

    # -------------------------
    # OBSERVATION BUILDER
    # -------------------------
    def _build_observation(self, reward, message) -> PatientObservation:
        case = self._current_case

        return PatientObservation(
            patient_id=case["patient_id"],
            age=case["age"],
            chief_complaint=case["chief_complaint"],
            vitals=case["vitals"],
            symptoms=case["symptoms"],
            medical_history=case["medical_history"],
            arrival_mode=case["arrival_mode"],
            time_in_waiting_room_minutes=case["time_in_waiting_room_minutes"],
            queue_length=case["queue_length"],
            task_id=self._state.task_id,
            done=self._done,
            reward=reward,
            masked=case.get("masked", False),
            attempts_remaining=self._remaining,
            message=message
        )

    # -------------------------
    # REWARD FUNCTION (0 → 1)
    # -------------------------
    def _compute_reward(self, action: TriageAction):
        esi = action.triage_level
        pathway = action.care_pathway
        confidence = action.confidence

        target_esi = self._target_esi
        target_pathway = self._target_pathway
        task = self._state.task_id

        esi_diff = abs(esi - target_esi)
        pathway_correct = (pathway == target_pathway)

        # Base reward (ESI accuracy)
        if esi_diff == 0:
            reward = 0.7
            message = "Correct ESI."
            correct = True
        elif esi_diff == 1:
            reward = 0.4
            message = f"Close (off by 1). Target ESI {target_esi}."
            correct = False
        elif esi_diff == 2:
            reward = 0.2
            message = f"Far (off by 2). Target ESI {target_esi}."
            correct = False
        else:
            reward = 0.0
            message = f"Incorrect. Target ESI {target_esi}."
            correct = False

        # Pathway bonus
        if pathway_correct and esi_diff == 0:
            reward += 0.2

        # Confidence bonus 
        if correct:
            reward += 0.1 * confidence
        else:
            reward -= 0.05 * confidence

        # Safety penalties
        if target_esi in [1, 2] and esi in [4, 5]:
            reward = 0.0
            message = "Critical under-triage! Dangerous decision."
            correct = True

        # Clamp reward
        reward = max(0.0, min(1.0, reward))

        return reward, message, correct

    # -------------------------
    # STATE (REQUIRED)
    # -------------------------
    @property
    def state(self) -> ERTriageState:
        return self._state