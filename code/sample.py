import json
import random
import os

files = [
    f for f in os.listdir("example-data/parcel/all-parcels")
    if os.path.isfile(os.path.join("example-data/parcel/all-parcels", f))
]

for file in files:
    # --- File paths ---
    input_file = "example-data/parcel/all-parcels/" + file
    output_file = "example-data/parcel/sampled-parcels/15pct_" + file

    # --- Load GeoJSON ---
    with open(input_file, "r") as f:
        data = json.load(f)

    features = data["features"]

    # --- Extract unique parcel IDs ---
    parcel_ids = set()
    for feature in features:
        props = feature.get("properties", {})
        pid = props.get("parcel_id")
        if pid is not None:
            parcel_ids.add(pid)

    parcel_ids = list(parcel_ids)

    # --- Sample 15% of unique parcel IDs ---
    sample_size = max(1, int(len(parcel_ids) * 0.15))  # ensure at least 1
    sampled_ids = set(random.sample(parcel_ids, sample_size))

    # --- Filter features ---
    filtered_features = [
        feature for feature in features
        if feature.get("properties", {}).get("parcel_id") in sampled_ids
    ]

    # --- Create output GeoJSON ---
    output_data = {
        "type": "FeatureCollection",
        "features": filtered_features
    }

    # --- Save result ---
    with open(output_file, "w") as f:
        json.dump(output_data, f)

    print(f"Total unique parcel_ids: {len(parcel_ids)}")
    print(f"Sampled parcel_ids: {len(sampled_ids)}")
    print(f"Output features: {len(filtered_features)}")