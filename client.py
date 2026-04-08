from openenv.core.env_client import EnvClient, StepResult
from models import TriageAction, PatientObservation, ERTriageState

class ERTriageClient(EnvClient):

    def _step_payload(self, action: TriageAction) -> dict:
        return action.model_dump()

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", payload)
        obs = PatientObservation(
            patient_id=obs_data.get("patient_id", ""),
            age=obs_data.get("age", 0),
            chief_complaint=obs_data.get("chief_complaint", ""),
            vitals=obs_data.get("vitals", {}),
            symptoms=obs_data.get("symptoms", []),
            medical_history=obs_data.get("medical_history", []),
            arrival_mode=obs_data.get("arrival_mode", ""),
            time_in_waiting_room_minutes=obs_data.get(
                "time_in_waiting_room_minutes", 0),
            queue_length=obs_data.get("queue_length", 0),
            task_id=obs_data.get("task_id", ""),
            done=obs_data.get("done", False),
            reward=obs_data.get("reward", 0.0),
            masked=obs_data.get("masked", False),
            attempts_remaining=obs_data.get("attempts_remaining", 0),
            message=obs_data.get("message", "")
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False)
        )

    def _parse_state(self, payload: dict) -> ERTriageState:
        return ERTriageState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            current_patient_id=payload.get("current_patient_id", ""),
            task_id=payload.get("task_id", ""),
            max_attempts=payload.get("max_attempts", 10)
        )
