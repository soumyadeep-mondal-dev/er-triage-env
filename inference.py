import os
import json
import asyncio
from openai import OpenAI
from client import ERTriageClient
from models import TriageAction

# ─────────────────────────────────────────────
# SAFE ENV VARIABLES (NO CRASHES)
# ─────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or ""
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://localhost:8000")

# ─────────────────────────────────────────────
# OPENAI CLIENT (SAFE INIT)
# ─────────────────────────────────────────────
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=API_KEY
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TASKS = ["classic_presentations", "ambiguous_cases", "masked_presentations"]
CASES_PER_TASK = 8

# ─────────────────────────────────────────────
# RULE-BASED FALLBACK (ALWAYS WORKS)
# ─────────────────────────────────────────────
def rule_based_action(obs):
    text = (obs.chief_complaint or "").lower()

    if "cardiac" in text or "arrest" in text:
        return 1, "resuscitation", 0.9

    if "chest pain" in text:
        return 2, "acute", 0.8

    if "fever" in text or "infection" in text:
        return 3, "observation", 0.7

    if "minor" in text:
        return 4, "fast_track", 0.6

    return 3, "observation", 0.5

# ─────────────────────────────────────────────
# PROMPT
# ─────────────────────────────────────────────
def build_prompt(obs):
    return f"""You are an expert emergency room triage nurse.
Assign the correct ESI triage level (1-5) and care pathway.

Patient:
Age: {obs.age}
Complaint: {obs.chief_complaint}
Vitals: {obs.vitals}
Symptoms: {', '.join(obs.symptoms)}

Respond in JSON:
{{"triage_level": <1-5>, "care_pathway": "<resuscitation|acute|fast_track|observation|discharge_likely>", "confidence": <0-1>}}"""

# ─────────────────────────────────────────────
# TASK RUNNER
# ─────────────────────────────────────────────
async def run_task(env_client, task_id):
    scores = []

    for i in range(CASES_PER_TASK):
        try:
            reset_result = await env_client.reset(seed=i, task_id=task_id)
            obs = reset_result.observation
        except Exception as e:
            print(f"[Reset Error] {e}")
            scores.append(0.0)
            continue

        for attempt in range(3):
            try:
                # ── TRY LLM (SAFE) ──
                if API_KEY:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[{"role": "user", "content": build_prompt(obs)}],
                        max_tokens=120,
                        temperature=0.0
                    )

                    raw = (response.choices[0].message.content or "").strip()

                    try:
                        parsed = json.loads(raw)
                    except Exception:
                        parsed = {}

                    triage_level = int(parsed.get("triage_level", 3))
                    pathway = parsed.get("care_pathway", "observation")
                    confidence = float(parsed.get("confidence", 0.5))

                else:
                    raise Exception("No API key → using fallback")

            except Exception as e:
                # ── FALLBACK (CRITICAL FOR VALIDATOR) ──
                print(f"[LLM Fallback] {e}")
                triage_level, pathway, confidence = rule_based_action(obs)

            try:
                result = await env_client.step(
                    TriageAction(
                        triage_level=triage_level,
                        care_pathway=pathway,
                        confidence=confidence
                    )
                )

                reward = max(float(result.reward or 0.0), 0.0)
                scores.append(reward)
                break

            except Exception as e:
                print(f"[Step Error] attempt {attempt}: {e}")
                if attempt == 2:
                    scores.append(0.0)

    avg_score = round(sum(scores) / len(scores), 4) if scores else 0.0

    return {
        "task_id": task_id,
        "score": avg_score,
        "cases_run": len(scores)
    }

# ─────────────────────────────────────────────
# MAIN (FAIL-SAFE)
# ─────────────────────────────────────────────
async def main():
    results = {}

    try:
        env_client = ERTriageClient(base_url=ENV_BASE_URL)

        for task in TASKS:
            print(f"\nRunning task: {task}")
            result = await run_task(env_client, task)
            results[task] = result
            print(f"Score: {result['score']}")

        overall = round(
            sum(r["score"] for r in results.values()) / len(results), 4
        )

        results["overall_score"] = overall

        print(f"\nOverall Score: {overall}")

    except Exception as e:
        # ── FINAL SAFETY NET ──
        print(f"[FATAL ERROR] {e}")
        results["overall_score"] = 0.0

    # ALWAYS SAVE OUTPUT (MANDATORY)
    try:
        with open("baseline_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Saved to baseline_results.json")
    except Exception as e:
        print(f"[Save Error] {e}")

# ─────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())