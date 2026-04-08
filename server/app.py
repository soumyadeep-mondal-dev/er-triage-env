from openenv.core.env_server import create_fastapi_app
from server.environment import ERTriageEnvironment
from models import TriageAction, PatientObservation

app = create_fastapi_app(
    ERTriageEnvironment,
    action_cls=TriageAction,
    observation_cls=PatientObservation
)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
