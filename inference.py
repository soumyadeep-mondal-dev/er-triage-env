import asyncio
import os
from typing import List, Optional

from openai import OpenAI
from client import ERTriageClient
from models import TriageAction


# ===== CONFIG =====
TASKS = [
    "classic_presentations",
    "ambiguous_cases",
    "masked_presentations",
]

MAX_STEPS = 5
ENV_BASE_URL = "https://deep-thinker-er-triage-env.hf.space"

API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
HF_TOKEN = os.getenv("HF_TOKEN")

BENCHMARK = "er_triage_env"


# ===== SAFE OPENAI CLIENT =====
try:
    # Use API_KEY dynamically if injected by grading platform, fallback to HF_TOKEN
    API_KEY = os.getenv("API_KEY") or HF_TOKEN
    if API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    else:
        client = None
except Exception:
    client = None


# ===== LOGGING =====
def log_start(task: str):
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]):
    error_val = error if error else "null"
    done_val = str(done).lower()

    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ===== ACTION LOGIC =====
def choose_action(obs) -> dict:
    try:
        triage_level = 1  # yellow
        care_pathway = "general"
        confidence = 0.7

        if client:
            import json
            import re
            prompt = f"Given the patient observation: {obs}, what is the triage level (int 1-5), care pathway (string), and confidence (float 0.0-1.0)? Return JSON with keys 'triage_level', 'care_pathway', 'confidence'."
            
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            res_content = response.choices[0].message.content
            json_match = re.search(r'\{.*\}', res_content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group(0))
                triage_level = data.get("triage_level", triage_level)
                care_pathway = data.get("care_pathway", care_pathway)
                confidence = data.get("confidence", confidence)
        else:
            if hasattr(obs, "symptoms") and "chest pain" in str(obs.symptoms).lower():
                triage_level = 3  # red
                care_pathway = "cardiac"
                confidence = 0.9

            elif hasattr(obs, "vitals") and obs.vitals.get("heart_rate", 0) > 110:
                triage_level = 2  # orange
                care_pathway = "urgent"
                confidence = 0.8

        return {
            "triage_level": int(triage_level),
            "care_pathway": str(care_pathway),
            "confidence": float(confidence),
        }

    except Exception:
        return {
            "triage_level": 1,
            "care_pathway": "general",
            "confidence": 0.5,
        }


# ===== TASK EXECUTION =====
async def run_task(env_client, task_id: str):
    rewards = []
    steps_taken = 0
    success = False

    log_start(task_id)

    try:
        try:
            result = await env_client.reset(task_id=task_id)
            obs = result.observation
        except Exception:
            log_end(False, 0, 0.0, [])
            return 0.0

        for step in range(1, MAX_STEPS + 1):
            steps_taken = step

            # ✅ ALWAYS define action_dict
            action_dict = {
                "triage_level": 1,
                "care_pathway": "general",
                "confidence": 0.5,
            }

            try:
                action_dict = choose_action(obs)
                action = TriageAction(**action_dict)

                result = await env_client.step(action)

                obs = result.observation
                reward = result.reward or 0.0
                done = result.done
                error = None

            except Exception as e:
                reward = 0.0
                done = False
                error = str(e)

            rewards.append(reward)

            log_step(
                step=step,
                action=str(action_dict["triage_level"]),  # ALWAYS SAFE
                reward=reward,
                done=done,
                error=error,
            )

            if done:
                break

        score = sum(rewards) / max(len(rewards), 1)
        score = max(0.0, min(score, 1.0))

        success = score > 0.3

    except Exception:
        score = 0.0

    finally:
        try:
            if hasattr(env_client, "close"):
                if asyncio.iscoroutinefunction(env_client.close):
                    await env_client.close()
                else:
                    env_client.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error: {e}", flush=True)
            
        log_end(success, steps_taken, score, rewards)

    return score


# ===== MAIN =====
async def main():
    env_client = ERTriageClient(base_url=ENV_BASE_URL)

    for task in TASKS:
        await run_task(env_client, task)


if __name__ == "__main__":
    asyncio.run(main())