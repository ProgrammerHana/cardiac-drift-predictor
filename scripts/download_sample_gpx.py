import urllib.request

# Real running activity with Garmin heart-rate trackpoint extensions
url = "https://raw.githubusercontent.com/juanifioren/gpx-converter/master/tests/assets/test.gpx"
output_file = "sample_run_with_hr.gpx"

print("Downloading run with heart rate data...")
urllib.request.urlretrieve(url, output_file)
print(f"Saved as: {output_file}")