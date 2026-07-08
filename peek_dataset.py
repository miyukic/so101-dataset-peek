# coding: UTF-8
import json
import sys
import pandas as pd

# python標準出力が特殊環境でデフォルトでcp932になるのを阻止
sys.stdout.reconfigure(encoding="utf-8")

# observation.state 今この瞬間、腕はこの角度にいる。
# action この瞬間、人間の操作では次にこの角度へ動かそうとしている。

PARQUET_PATH =\
    r".\data\orsoromeo_so101_pick_and_place\data\chunk-000\episode_000000.parquet"
INFO_JSON_PATH =\
    r".\data\orsoromeo_so101_pick_and_place\meta\info.json"


def outputInfoJson():
    with open(INFO_JSON_PATH) as file:
        jsonfile = json.load(file)

    keys = ["robot_type", "total_episodes", "total_frames", "fps", "features"]
    print("=== Dataset Overview ===")
    for i in keys:
        if i == "features":
            print(f"=== Key {i} のKeyリスト===")
            for featuresKey in jsonfile[i].keys():
                print(f"  - {featuresKey}")
            continue
        print(f"  {i}: {jsonfile[i]}")


def outputParquet():
    print("=== First Parquet ===")
    df = pd.read_parquet(PARQUET_PATH)
    print(f"  shape: {df.shape}")
    print("=== columns と データ型 ===")
    for i in df.columns:
        print(f"  - {i}: {df[i].dtype}")


def outputFirstRow():
    print("=== First Row ===")
    df = pd.read_parquet(PARQUET_PATH)
    for i in df.columns:
        print(f"  - {i}: {df.iloc[0][i]}")
    # print(df.iloc[0])

    print("=== actionとobservation.state の 次元数 ===")
    print(f"  - action次元: {len(df.iloc[0]['action'])}")
    print(f"  - observation.state次元: {len(df.iloc[0]['observation.state'])}")


def outputFirstRowAction():
    print("=== First Row Action ===")
    df = pd.read_parquet(PARQUET_PATH)
    actionlist =\
    [
        "main_shoulder_pan",
        "main_shoulder_lift",
        "main_elbow_flex",
        "main_wrist_flex",
        "main_wrist_roll",
        "main_gripper"
    ]
    for n, i in enumerate(actionlist):
        print(f"  - {i}: {df.iloc[0]['action'][n]}")


def outputFirstRowObservationState():
    print("=== First Row Observation State ===")
    df = pd.read_parquet(PARQUET_PATH)
    actionlist =\
    [
        "main_shoulder_pan",
        "main_shoulder_lift",
        "main_elbow_flex",
        "main_wrist_flex",
        "main_wrist_roll",
        "main_gripper"
    ]
    for n, i in enumerate(actionlist):
        print(f"  - {i}: {df.iloc[0]['observation.state'][n]}")



def main():
    outputInfoJson()
    outputParquet()
    outputFirstRow()

    # 現在の値
    outputFirstRowObservationState()

    # 人間による操作の次の値（この値に変わろうとしている）
    outputFirstRowAction()


if __name__ == "__main__":
    main()

