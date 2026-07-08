# coding: UTF-8
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

EPISODE_INDEX = 0
START_FRAME = 100
CHUNK_SIZE = 16
DATASET_ROOT = Path(r".\data\orsoromeo_so101_pick_and_place")

JOINT_NAMES = [
    "main_shoulder_pan",
    "main_shoulder_lift",
    "main_elbow_flex",
    "main_wrist_flex",
    "main_wrist_roll",
    "main_gripper",
]


def episodeName():
    return f"episode_{EPISODE_INDEX:06d}"


def parquetPath():
    return DATASET_ROOT / "data" / "chunk-000" / f"{episodeName()}.parquet"


def printJointValues(title, values):
    print(title)
    for name, value in zip(JOINT_NAMES, values):
        print(f"  - {name}: {value}")


def main():
    df = pd.read_parquet(parquetPath())
    endFrame = min(START_FRAME + CHUNK_SIZE, len(df))
    rows = df.iloc[START_FRAME:endFrame]
    actionChunk = list(rows["action"])

    print("=== Action Chunk ===")
    print(f"  episode     : {EPISODE_INDEX}")
    print(f"  start_frame : {START_FRAME}")
    print(f"  chunk_size  : {len(actionChunk)}")
    print(f"  shape       : {len(actionChunk)} x {len(actionChunk[0])}")
    print("  meaning     : target actions from the current frame into the future")

    startRow = df.iloc[START_FRAME]
    print()
    printJointValues("=== Current Observation State ===", startRow["observation.state"])

    print("\n=== Future Actions ===")
    for row in rows.itertuples():
        action = getattr(row, "action")
        frameIndex = getattr(row, "frame_index")
        timestamp = getattr(row, "timestamp")
        values = ", ".join(f"{name}={value:.3f}" for name, value in zip(JOINT_NAMES, action))
        print(f"  frame={frameIndex:03d}  time={timestamp:.3f}s  {values}")

    firstAction = actionChunk[0]
    lastAction = actionChunk[-1]
    delta = [last - first for first, last in zip(firstAction, lastAction)]
    print()
    printJointValues("=== Last Action - First Action ===", delta)


if __name__ == "__main__":
    main()
