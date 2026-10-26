import ast
import math
from pathlib import Path
import numpy as np
import pandas as pd

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def extract_run_features(record):
    hr_list = record.get("heart_rate", [])
    time_list = record.get("timestamp", [])
    alt_list = record.get("altitude", [])
    speed_list = record.get("speed", [])
    lat_list = record.get("latitude", [])
    lon_list = record.get("longitude", [])

    n_points = len(hr_list)
    if n_points < 300 or len(time_list) < 300:
        return None

    time_arr = np.array(time_list, dtype=float)
    hr_arr = np.array(hr_list, dtype=float)
    total_duration_s = time_arr[-1] - time_arr[0]

    if total_duration_s < 1800:
        return None

    # 1. תיקון יחידות המהירות הקריטי: הכל מומר ל-km/h באופן אחיד
    if len(speed_list) == n_points:
        speed_arr = np.array(speed_list, dtype=float)
    elif len(lat_list) == n_points and len(lon_list) == n_points:
        speeds = [0.0]
        for i in range(1, n_points):
            dt = time_arr[i] - time_arr[i - 1]
            if dt > 0:
                dist = haversine_distance(lat_list[i - 1], lon_list[i - 1], lat_list[i], lon_list[i])
                speeds.append((dist / dt) * 3.6) # המרה מ-m/s ל-km/h
            else:
                speeds.append(0.0)
        speed_arr = np.array(speeds, dtype=float)
    else:
        return None

    rel_time_s = time_arr - time_arr[0]
    valid_running = (speed_arr > 6.5) & (hr_arr > 80) # סף ריצה הגיוני ודופק מינימלי

    # --- חילוץ מאפיינים מדקות 5 עד 15 (לאחר חימום) ---
    mask_5_15 = (rel_time_s >= 300) & (rel_time_s <= 900) & valid_running
    if np.sum(mask_5_15) < 30:
        return None

    time_5_15 = rel_time_s[mask_5_15]
    hr_5_15 = hr_arr[mask_5_15]
    speed_5_15 = speed_arr[mask_5_15]

    hr_mean_5_15 = float(np.mean(hr_5_15))
    speed_mean_5_15 = float(np.mean(speed_5_15))
    initial_ratio = hr_mean_5_15 / speed_mean_5_15

    hr_slope_5_15, _ = np.polyfit(time_5_15 / 60.0, hr_5_15, 1)
    speed_slope_5_15, _ = np.polyfit(time_5_15 / 60.0, speed_5_15, 1)
    ratio_slope_5_15, _ = np.polyfit(time_5_15 / 60.0, hr_5_15 / speed_5_15, 1)

    mask_5_10 = (rel_time_s >= 300) & (rel_time_s <= 600) & valid_running
    mask_10_15 = (rel_time_s > 600) & (rel_time_s <= 900) & valid_running
    
    if np.sum(mask_5_10) > 15 and np.sum(mask_10_15) > 15:
        ratio_5_10 = np.mean(hr_arr[mask_5_10]) / np.mean(speed_arr[mask_5_10])
        ratio_10_15 = np.mean(hr_arr[mask_10_15]) / np.mean(speed_arr[mask_10_15])
        early_decoupling_pct = ((ratio_10_15 / ratio_5_10) - 1.0) * 100.0
    else:
        early_decoupling_pct = 0.0

    if len(alt_list) == n_points:
        alt_15m = pd.Series(alt_list)[(rel_time_s <= 900)].rolling(window=5, min_periods=1).mean().values
        diffs = np.diff(alt_15m)
        elev_gain_15m = float(np.sum(diffs[diffs > 0]))
    else:
        elev_gain_15m = 0.0

    speed_std_15m = float(np.std(speed_5_15))

    half_time = rel_time_s[0] + (total_duration_s / 2.0)
    mask_first_half = (rel_time_s >= 300) & (rel_time_s <= half_time) & valid_running
    mask_second_half = (rel_time_s > half_time) & valid_running

    if np.sum(mask_first_half) < 60 or np.sum(mask_second_half) < 60:
        return None

    # חישוב מתוקן - יחס הממוצעים
    ratio_1 = np.mean(hr_arr[mask_first_half]) / np.mean(speed_arr[mask_first_half])
    ratio_2 = np.mean(hr_arr[mask_second_half]) / np.mean(speed_arr[mask_second_half])
    target_drift_pct = float(((ratio_2 / ratio_1) - 1.0) * 100.0)

    if target_drift_pct < -15.0 or target_drift_pct > 40.0:
        return None

    return {
        "user_id": record.get("userId"),
        "gender": 1 if record.get("gender") == "male" else 0,
        "total_duration_min": round(total_duration_s / 60.0, 2),
        "hr_mean_5_15": round(hr_mean_5_15, 2),
        "speed_mean_5_15": round(speed_mean_5_15, 2),
        "initial_ratio": round(initial_ratio, 4),
        "hr_slope_5_15": round(hr_slope_5_15, 4),
        "speed_slope_5_15": round(speed_slope_5_15, 4),
        "ratio_slope_5_15": round(ratio_slope_5_15, 4),
        "early_decoupling_pct": round(early_decoupling_pct, 4),
        "elev_gain_15m": round(elev_gain_15m, 2),
        "speed_std_15m": round(speed_std_15m, 4),
        "target_drift_pct": round(target_drift_pct, 4),
    }

def main():
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    data_input = PROJECT_ROOT / "data" / "raw" / "endomondoHR.json"
    output_dir = PROJECT_ROOT / "data" / "processed"
    output_path = output_dir / "dataset_sample.csv"

    print(f"Reading raw data from: {data_input}")
    
    target_total_runs = 2000
    max_runs_per_user = 5
    user_run_counts = {}
    processed_runs = []

    with open(data_input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            record = ast.literal_eval(line)
            user_id = record.get("userId")

            if user_run_counts.get(user_id, 0) >= max_runs_per_user:
                continue

            if record.get("sport") == "run" and "heart_rate" in record:
                feats = extract_run_features(record)
                if feats:
                    processed_runs.append(feats)
                    user_run_counts[user_id] = user_run_counts.get(user_id, 0) + 1
                    
                    if len(processed_runs) % 200 == 0:
                        print(f"Collected {len(processed_runs)} runs...")
                        
                if len(processed_runs) >= target_total_runs:
                    break

    df_dataset = pd.DataFrame(processed_runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_dataset.to_csv(output_path, index=False)
    
    print("\nExtraction finished!")
    print(f"Shape: {df_dataset.shape}")
    print(df_dataset.describe())

if __name__ == "__main__":
    main()