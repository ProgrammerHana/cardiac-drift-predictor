import ast
from pathlib import Path

# שורש הפרויקט: תיקייה אחת מעל scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_path = PROJECT_ROOT / "data" / "raw" / "endomondoHR.json"

print(f"Looking for dataset at: {data_path.resolve()}")

if not data_path.exists():
    raise FileNotFoundError(f"Could not find endomondoHR.json at {data_path.resolve()}")

running_workout = None

print("Scanning for the first running workout...")
with open(data_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f):
        line = line.strip()
        if not line:
            continue
        
        record = ast.literal_eval(line)
        
        if record.get("sport") == "run" and "heart_rate" in record:
            running_workout = record
            print(f"Found running workout at line {line_idx}!")
            break

if running_workout:
    print("\n--- Workout Summary ---")
    print(f"User ID: {running_workout.get('userId')}")
    print(f"Sport: {running_workout.get('sport')}")
    print(f"Gender: {running_workout.get('gender')}")
    
    timestamps = running_workout.get("timestamp", [])
    hr = running_workout.get("heart_rate", [])
    speed = running_workout.get("speed", [])
    altitude = running_workout.get("altitude", [])
    
    print(f"Total timepoints: {len(timestamps)}")
    print(f"Sample HR values: {hr[:5]}")
    print(f"Sample Speed values: {speed[:5]}")
    print(f"Sample Altitude values: {altitude[:5]}")
else:
    print("No running workout with heart rate found.")