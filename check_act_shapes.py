# coding: UTF-8
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

DATASET_ROOT = Path(r".\data\orsoromeo_so101_pick_and_place")
INFO_JSON_PATH = DATASET_ROOT / "meta" / "info.json"
PARQUET_PATH = DATASET_ROOT / "data" / "chunk-000" / "episode_000000.parquet"

CAMERA_KEYS = [
    "observation.images.laptop",
    "observation.images.phone",
]

STATE_KEY = "observation.state"
ACTION_KEY = "action"


def shapeText(shape):
    return " x ".join(str(i) for i in shape)


def valueShape(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    return [len(value)]


def printFeatureShape(features, key, role):
    feature = features[key]
    print(f"  - {key}")
    print(f"      role : {role}")
    print(f"      dtype: {feature['dtype']}")
    print(f"      shape: {shapeText(feature['shape'])}")


def main():
    info = json.loads(INFO_JSON_PATH.read_text(encoding="utf-8"))
    features = info["features"]
    df = pd.read_parquet(PARQUET_PATH)
    row = df.iloc[0]

    print("=== Dataset ===")
    print(f"  robot_type    : {info['robot_type']}")
    print(f"  total_episodes: {info['total_episodes']}")
    print(f"  total_frames  : {info['total_frames']}")
    print(f"  fps           : {info['fps']}")

    print("\n=== ACT Inputs ===")
    for cameraKey in CAMERA_KEYS:
        printFeatureShape(features, cameraKey, "camera image")
    printFeatureShape(features, STATE_KEY, "robot current joint state")

    print("\n=== ACT Target ===")
    printFeatureShape(features, ACTION_KEY, "next joint command")

    print("\n=== First Parquet Check ===")
    print(f"  parquet path : {PARQUET_PATH}")
    print(f"  episode rows : {df.shape[0]}")
    print(f"  columns      : {df.shape[1]}")
    print(f"  state shape  : {shapeText(valueShape(row[STATE_KEY]))}")
    print(f"  action shape : {shapeText(valueShape(row[ACTION_KEY]))}")

    print("\n=== Shape Summary ===")
    laptopShape = shapeText(features["observation.images.laptop"]["shape"])
    phoneShape = shapeText(features["observation.images.phone"]["shape"])
    stateShape = shapeText(features[STATE_KEY]["shape"])
    actionShape = shapeText(features[ACTION_KEY]["shape"])
    print(f"  input : laptop image {laptopShape}")
    print(f"        + phone image  {phoneShape}")
    print(f"        + state        {stateShape}")
    print(f"  target: action       {actionShape}")
    print("  unit  : one row = one frame")


if __name__ == "__main__":
    main()
