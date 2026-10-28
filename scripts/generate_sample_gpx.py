import math
from datetime import datetime, timedelta


def generate_running_gpx(filename="sample_run_with_hr.gpx", duration_minutes=40):
    start_time = datetime(2026, 10, 17, 7, 0, 0)
    total_seconds = duration_minutes * 60

    header = """<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="PythonGPX"
  xmlns="http://www.topografix.com/GPX/1/1"
  xmlns:gpxtpx="http://www.garmin.com/xmlschemas/TrackPointExtension/v1">
  <trk>
    <name>Morning Run</name>
    <trkseg>
"""

    footer = """    </trkseg>
  </trk>
</gpx>
"""

    points = []
    base_lat = 32.0853
    base_lon = 34.7818
    base_ele = 20.0

    for s in range(total_seconds):
        t = start_time + timedelta(seconds=s)
        time_str = t.strftime("%Y-%m-%dT%H:%M:%SZ")

        lat = base_lat + (s * 0.000028)
        lon = base_lon + (s * 0.000008)
        ele = base_ele + 3.0 * math.sin(s / 120.0)

        drift = (s / total_seconds) * 16.0
        noise = 2.0 * math.sin(s / 15.0)
        hr = int(140 + drift + noise)

        pt = f"""      <trkpt lat="{lat:.6f}" lon="{lon:.6f}">
        <ele>{ele:.1f}</ele>
        <time>{time_str}</time>
        <extensions>
          <gpxtpx:TrackPointExtension>
            <gpxtpx:hr>{hr}</gpxtpx:hr>
          </gpxtpx:TrackPointExtension>
        </extensions>
      </trkpt>"""
        points.append(pt)

    with open(filename, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(points) + "\n" + footer)

    print(f"File created successfully: {filename} ({total_seconds} data points)")


if __name__ == "__main__":
    generate_running_gpx()
