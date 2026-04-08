import random
import json
import os

random.seed(42)

def generate_classic_cases(start_id, count=50):
    cases = []
    for i in range(count):
        esi = random.randint(1, 4) if i < 40 else 5 # Ensuring mix of cases
        pathway_map = {1: "resuscitation", 2: "acute", 3: "acute", 4: "fast_track", 5: "discharge_likely"}
        
        case = {
            "patient_id": f"case_{start_id+i:03d}",
            "task_id": "classic_presentations",
            "age": random.randint(18, 65),
            "vitals": {
                "heart_rate": 80,
                "bp_systolic": 120,
                "bp_diastolic": 80,
                "respiratory_rate": 16,
                "spo2": 98,
                "temperature": 37.0,
                "gcs_score": 15
            },
            "symptoms": [],
            "medical_history": [],
            "arrival_mode": random.choice(["ambulance", "walk-in"]),
            "time_in_waiting_room_minutes": 0 if esi <= 2 else random.randint(30, 120),
            "queue_length": random.randint(0, 10),
            "masked": False,
            "ground_truth_esi": esi,
            "ground_truth_pathway": pathway_map[esi]
        }
        
        if esi == 1:
            case["chief_complaint"] = "unresponsive"
            case["vitals"]["heart_rate"] = random.choice([0, 40, 150])
            case["vitals"]["bp_systolic"] = 60
            case["vitals"]["spo2"] = 80
            case["vitals"]["gcs_score"] = 3
            case["symptoms"] = ["cyanosis", "apnea"]
            case["arrival_mode"] = "ambulance"
            case["ground_truth_pathway"] = "resuscitation"
            
        elif esi == 2:
            case["chief_complaint"] = "chest pain"
            case["vitals"]["heart_rate"] = 102
            case["vitals"]["bp_systolic"] = 88
            case["vitals"]["bp_diastolic"] = 60
            case["vitals"]["respiratory_rate"] = 22
            case["vitals"]["spo2"] = 94
            case["symptoms"] = ["diaphoresis", "jaw pain", "left arm radiation"]
            case["medical_history"] = ["hypertension", "hyperlipidemia"]
            case["arrival_mode"] = "ambulance"
            # Setting path to resuscitation matching the example in instructions
            if i % 2 == 0:
                case["ground_truth_pathway"] = "resuscitation"
                case["ground_truth_esi"] = 1
            
        elif esi == 3:
            case["chief_complaint"] = "abdominal pain"
            case["vitals"]["temperature"] = 38.5
            case["symptoms"] = ["nausea", "vomiting", "RLQ tenderness"]
            
        elif esi == 4:
            case["chief_complaint"] = "ankle sprain"
            case["symptoms"] = ["swelling", "pain with weight bearing"]
            
        elif esi == 5:
            case["chief_complaint"] = "medication refill"
            case["symptoms"] = ["none"]
            
        cases.append(case)
    return cases

def generate_ambiguous_cases(start_id, count=50):
    cases = []
    # Mix of conflicting vitals/symptoms, elderly patients
    for i in range(count):
        esi = random.choice([2, 3]) 
        pathway_map = {1: "resuscitation", 2: "acute", 3: "acute", 4: "fast_track", 5: "discharge_likely"}
        
        case = {
            "patient_id": f"case_{start_id+i:03d}",
            "task_id": "ambiguous_cases",
            "age": random.randint(70, 95),
            "vitals": {
                "heart_rate": random.randint(85, 110),
                "bp_systolic": random.randint(90, 160),
                "bp_diastolic": random.randint(50, 90),
                "respiratory_rate": random.randint(18, 24),
                "spo2": random.randint(92, 95),
                "temperature": round(random.uniform(36.0, 37.5), 1),
                "gcs_score": random.choice([14, 15])
            },
            "symptoms": ["weakness", "dizziness", "mild confusion"],
            "medical_history": ["dementia", "hypertension", "diabetes", "heart failure"],
            "arrival_mode": "walk-in",
            "time_in_waiting_room_minutes": random.randint(10, 45),
            "queue_length": random.randint(5, 15),
            "masked": False,
            "ground_truth_esi": esi,
            "ground_truth_pathway": pathway_map[esi]
        }
        if esi == 2:
            case["chief_complaint"] = "fall, on blood thinners"
            case["symptoms"].append("head strike")
        else:
            case["chief_complaint"] = "general weakness"

        cases.append(case)
    return cases
    
def generate_masked_cases(start_id, count=50):
    cases = []
    # silent MI, afebrile sepsis, stroke mimics. masked=True hides some vitals
    for i in range(count):
        case_type = random.choice(["silent_mi", "afebrile_sepsis", "stroke_mimic"])
        
        case = {
            "patient_id": f"case_{start_id+i:03d}",
            "task_id": "masked_presentations",
            "age": random.randint(50, 80),
            "vitals": {
                "heart_rate": 90,
                "bp_systolic": 110,
                "bp_diastolic": 70,
                "respiratory_rate": 18,
                "spo2": 96,
                "temperature": 36.8,
                "gcs_score": 15
            },
            "symptoms": [],
            "medical_history": [],
            "arrival_mode": "ambulance",
            "time_in_waiting_room_minutes": 0,
            "queue_length": random.randint(2, 8),
            "masked": True,
            "ground_truth_esi": 2, # Most of these are high-risk (ESI 2) masquerading as lower risk
            "ground_truth_pathway": "acute"
        }
        
        if case_type == "silent_mi":
            case["chief_complaint"] = "indigestion and fatigue"
            case["medical_history"] = ["diabetes", "neuropathy"]
            case["symptoms"] = ["nausea", "sweating"]
            case["ground_truth_esi"] = 2
            case["ground_truth_pathway"] = "acute"
            
        elif case_type == "afebrile_sepsis":
            case["chief_complaint"] = "altered mental status"
            case["age"] = random.randint(80, 95)
            case["vitals"]["temperature"] = 36.1
            case["vitals"]["heart_rate"] = 115
            case["vitals"]["bp_systolic"] = 85
            case["medical_history"] = ["recent UTI", "immunosuppression"]
            case["ground_truth_esi"] = 2
            case["ground_truth_pathway"] = "resuscitation"
            
        elif case_type == "stroke_mimic":
            case["chief_complaint"] = "slurred speech and confusion"
            case["medical_history"] = ["seizure disorder"]
            case["symptoms"] = ["postictal state", "Todd's paresis"]
            case["ground_truth_esi"] = 2
            case["ground_truth_pathway"] = "acute"

        cases.append(case)
    return cases

if __name__ == "__main__":
    cases = []
    cases.extend(generate_classic_cases(1, 50))
    cases.extend(generate_ambiguous_cases(51, 50))
    cases.extend(generate_masked_cases(101, 50))
    
    # Needs to match server/cases.json path inside er-triage-env/
    out_path = os.path.join(os.path.dirname(__file__), "server", "cases.json")
    with open(out_path, "w") as f:
        json.dump(cases, f, indent=2)
    print(f"Generated {len(cases)} cases into {out_path}")
