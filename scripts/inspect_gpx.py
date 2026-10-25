import math
import pandas as pd
import gpxpy

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    מחשב מרחק במטרים בין שתי נקודות על כדור הארץ לפי קווי אורך ורוחב.
    """
    r = 6371000.0  # רדיוס כדור הארץ במטרים
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return r * c

def parse_gpx(file_path: str) -> pd.DataFrame:
    with open(file_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)

    records = []
    for track in gpx.tracks:
        for segment in track.segments:
            for point in segment.points:
                hr_val = None
                for ext in point.extensions:
                    for child in ext:
                        if "hr" in child.tag.lower():
                            try:
                                hr_val = float(child.text)
                            except (TypeError, ValueError):
                                pass

                records.append({
                    "timestamp": point.time,
                    "latitude": point.latitude,
                    "longitude": point.longitude,
                    "elevation": point.elevation,
                    "heart_rate": hr_val
                })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # המרת עמודת הזמן ומיון
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # 1. הפרש זמן בשניות בין כל נקודה לנקודה שקדמה לה
    df["time_diff_s"] = df["timestamp"].diff().dt.total_seconds().fillna(0)

    # 2. חישוב מרחק במטרים בין נקודות עוקבות
    prev_lat = df["latitude"].shift(1)
    prev_lon = df["longitude"].shift(1)

    distances = [0.0]
    for i in range(1, len(df)):
        d = haversine_distance(
            prev_lat.iloc[i], prev_lon.iloc[i],
            df["latitude"].iloc[i], df["longitude"].iloc[i]
        )
        distances.append(d)

    df["distance_m"] = distances
    df["cumulative_distance_m"] = df["distance_m"].cumsum()

    # 3. חישוב מהירות (מטר/שנייה) תוך הימנעות מחלוקה באפס
    df["speed_ms"] = 0.0
    valid_time = df["time_diff_s"] > 0
    df.loc[valid_time, "speed_ms"] = df.loc[valid_time, "distance_m"] / df.loc[valid_time, "time_diff_s"]

    return df

if __name__ == "__main__":
    df = parse_gpx("sample_run_with_hr.gpx")
    
    print("Shape:", df.shape)
    print("\nFirst 5 rows with telemetry:")
    print(df[["timestamp", "heart_rate", "speed_ms", "cumulative_distance_m"]].head())