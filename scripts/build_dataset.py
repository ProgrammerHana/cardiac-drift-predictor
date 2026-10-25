import ast
import math
from pathlib import Path
import numpy as np
import pandas as pd


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000  # רדיוס כדור הארץ במטרים
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
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

    # סינון: מינימום 30 דקות ריצה
    if total_duration_s < 1800:
        return None

    # חילוץ מהירות מנתוני מהירות קיימים או מחישוב מרחק GPS
    if len(speed_list) == n_points:
        speed_arr = np.array(speed_list, dtype=float)
    elif len(lat_list) == n_points and len(lon_list) == n_points:
        speeds = [0.0]
        for i in range(1, n_points):
            dt = time_arr[i] - time_arr[i - 1]
            if dt > 0:
                dist = haversine_distance(
                    lat_list[i - 1], lon_list[i - 1], lat_list[i], lon_list[i]
                )
                speeds.append(dist / dt)
            else:
                speeds.append(0.0)
        speed_arr = np.array(speeds, dtype=float)
    else:
        return None

    # יצירת ציר זמן יחסי בשניות מתחילת הריצה
    rel_time_s = time_arr - time_arr[0]

    # --- חילוץ מאפיינים מ-15 הדקות הראשונות (0 עד 900 שניות) ---
    mask_15m = rel_time_s <= 900
    if np.sum(mask_15m) < 60:
        return None

    hr_15m = hr_arr[mask_15m]
    speed_15m = speed_arr[mask_15m]
    time_15m = rel_time_s[mask_15m]

    # סינון עצירות ברמזורים ואי-דיוקי חיישן
    valid_15m = (speed_15m > 1.0) & (hr_15m > 60)
    if np.sum(valid_15m) < 40:
        return None

    clean_hr_15m = hr_15m[valid_15m]
    clean_speed_15m = speed_15m[valid_15m]

    initial_ratio = float(np.mean(clean_hr_15m / clean_speed_15m))

    # שיפוע דופק בדקות 5 עד 15 (300 עד 900 שניות)
    mask_5_15 = (time_15m >= 300) & (time_15m <= 900) & valid_15m
    if np.sum(mask_5_15) >= 20:
        slope, _ = np.polyfit(time_15m[mask_5_15] / 60.0, hr_15m[mask_5_15], 1)
        hr_slope_5_15 = float(slope)
    else:
        hr_slope_5_15 = 0.0

    # טיפוס מצטבר ב-15 דקות הראשונות
    if len(alt_list) == n_points:
        alt_15m = np.array(alt_list, dtype=float)[mask_15m]
        diffs = np.diff(alt_15m)
        elev_gain_15m = float(np.sum(diffs[diffs > 0]))
    else:
        elev_gain_15m = 0.0

    speed_std_15m = float(np.std(clean_speed_15m))

    # --- חישוב ה-Target על כל הריצה (חצי ראשון מול חצי שני) ---
    half_time = rel_time_s[0] + (total_duration_s / 2.0)
    mask_first_half = (rel_time_s <= half_time) & (speed_arr > 1.0) & (hr_arr > 60)
    mask_second_half = (rel_time_s > half_time) & (speed_arr > 1.0) & (hr_arr > 60)

    if np.sum(mask_first_half) < 60 or np.sum(mask_second_half) < 60:
        return None

    ratio_1 = np.mean(hr_arr[mask_first_half] / speed_arr[mask_first_half])
    ratio_2 = np.mean(hr_arr[mask_second_half] / speed_arr[mask_second_half])

    target_drift_pct = float(((ratio_2 / ratio_1) - 1.0) * 100.0)

    # סינון ערכי קיצון פיזיולוגיים לא הגיוניים
    if target_drift_pct < -25.0 or target_drift_pct > 60.0:
        return None

    gender_val = 1 if record.get("gender") == "male" else 0

    return {
        "user_id": record.get("userId"),
        "gender": gender_val,
        "total_duration_min": round(total_duration_s / 60.0, 2),
        "initial_ratio": round(initial_ratio, 4),
        "hr_slope_5_15": round(hr_slope_5_15, 4),
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
            if not line:
                continue

            record = ast.literal_eval(line)
            user_id = record.get("userId")

            if user_run_counts.get(user_id, 0) >= max_runs_per_user:
                continue

            if record.get("sport") == "run" and "heart_rate" in record:
                feats = extract_run_features(record)
                if feats:
                    processed_runs.append(feats)
                    user_run_counts[user_id] = (
                        user_run_counts.get(user_id, 0) + 1
                    )

                    if len(processed_runs) % 200 == 0:
                        print(
                            f"Collected {len(processed_runs)} runs across {len(user_run_counts)} runners..."
                        )

                if len(processed_runs) >= target_total_runs:
                    break

    df_dataset = pd.DataFrame(processed_runs)
    output_dir.mkdir(parents=True, exist_ok=True)
    df_dataset.to_csv(output_path, index=False)

    print("\nExtraction finished!")
    print(f"File saved: {output_path}")
    print(f"Shape: {df_dataset.shape}")
    print(f"Unique runners: {df_dataset['user_id'].nunique()}")
    print("\nSummary statistics:")
    print(df_dataset.describe())


if __name__ == "__main__":
    main()