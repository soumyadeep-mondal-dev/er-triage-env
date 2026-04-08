---
title: er-triage-env
emoji: 🏥
colorFrom: blue
colorTo: green
sdk: docker
---

# 🏥 ER Triage Decision Environment (OpenEnv)

## 📌 Overview

This project implements a **real-world reinforcement learning environment** simulating emergency room (ER) triage decisions.

The agent acts as a **triage nurse**, assigning:

* **ESI level (1–5)** → urgency classification
* **Care pathway** → treatment routing

The goal is to **maximize patient safety and triage accuracy**.

---

## 🎯 Problem Motivation

Emergency triage is a **high-stakes decision-making task**:

* Under-triage → life-threatening delays
* Over-triage → resource overload
* Ambiguous cases → require reasoning

This environment evaluates whether AI systems can:

* interpret patient data
* make safe decisions
* handle uncertainty

---

## 🧠 Environment Design

### 🔹 Observation Space

Each observation represents a patient:

```python
PatientObservation:
    patient_id: str
    age: int
    chief_complaint: str
    vitals: Dict[str, float]
    symptoms: List[str]
    medical_history: List[str]
    arrival_mode: str
    time_in_waiting_room_minutes: int
    queue_length: int
    task_id: str
    done: bool
    reward: float
    masked: bool
    attempts_remaining: int
    message: str
```

---

### 🔹 Action Space

```python
TriageAction:
    triage_level: int  # 1 (most urgent) to 5 (least urgent)
    care_pathway: str  # resuscitation | acute | fast_track | observation | discharge_likely
    confidence: float  # 0.0 to 1.0
```

---

### 🔹 State

```python
ERTriageState:
    episode_id: str
    step_count: int
    current_patient_id: str
    task_id: str
    max_attempts: int
```

---

## 🔁 Interaction Flow

```text
reset() → returns patient case
step(action) → returns reward + feedback
state → environment metadata
```

Each episode:

* One patient case
* Agent makes triage decision
* Reward assigned
* Episode ends

---

## 🎯 Tasks (Difficulty Levels)

The environment includes **3 task categories**:

### 🟢 1. Classic Presentations (Easy)

* Clear symptoms
* Obvious triage decisions
* Example: cardiac arrest, severe trauma

---

### 🟡 2. Ambiguous Cases (Medium)

* Mixed symptoms
* Requires reasoning
* Example: chest pain + anxiety

---

### 🔴 3. Masked Presentations (Hard)

* Hidden or misleading symptoms
* High risk of misclassification
* Example: atypical heart attack

---

## 📊 Reward Design (0.0 → 1.0)

The reward function is **continuous and safety-aware**.

### ✅ Base scoring:

* Correct ESI → **0.7**
* Off by 1 → **0.4**
* Off by 2 → **0.2**
* Incorrect → **0.0**

---

### ➕ Bonus:

* Correct care pathway → **+0.2**
* Confidence (if correct) → **+0.1 × confidence**

---

### ➖ Penalty:

* Incorrect confident decisions → **−0.05 × confidence**
* Critical under-triage → **0.0 (severe penalty)**

---

### 🔒 Final reward:

```text
Clamped between 0.0 and 1.0
```

---

## 🧪 Example Interaction

```python
obs = env.reset()

action = TriageAction(
    triage_level=2,
    care_pathway="acute",
    confidence=0.8
)

result = env.step(action)

print(result.reward)
print(result.message)
```

---

## ⚙️ Setup Instructions

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2. Run locally

```bash
uvicorn server.app:app --reload
```

Check:

```
http://localhost:8000/health
```

---

### 3. Run inference

```bash
export HF_TOKEN=your_token
export MODEL_NAME=your_model
export ENV_BASE_URL=http://localhost:8000

python inference.py
```

---

## 🐳 Docker

### Build:

```bash
docker build -t er-triage-env -f server/Dockerfile .
```

### Run:

```bash
docker run -p 8000:8000 er-triage-env
```

---

## 🌐 Deployment (Hugging Face)

```bash
openenv push --repo-id <username>/er-triage-env
```

Access:

```
https://<username>-er-triage-env.hf.space
```

---

## 📈 Evaluation

The system is evaluated on:

* Accuracy of ESI classification
* Correct care pathway
* Safety (avoiding under-triage)
* Robustness across tasks

Final score:

```text
Average across all tasks
```

---

## 🚀 Key Features

* ✅ Real-world healthcare scenario
* ✅ Continuous reward function (not binary)
* ✅ Multi-difficulty tasks
* ✅ Safety-aware penalties
* ✅ Type-safe design
* ✅ OpenEnv compliant

---

## ⚠️ Notes

* Ground truth labels are **never exposed** to the agent
* Masked tasks intentionally hide key signals
* Designed to challenge advanced LLM reasoning

---

## 🧠 Summary

This environment tests whether AI systems can make **safe, accurate, and context-aware medical triage decisions** under uncertainty.

---
