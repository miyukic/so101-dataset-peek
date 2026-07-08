# coding: UTF-8
import pandas as pd
import imageio
from pathlib import Path

EPISODE_INDEX = 0
FRAME_INDEX = 100  
DATASET_ROOT = r".\data\orsoromeo_so101_pick_and_place"

VIDEO_OUTPUT_PATH = r"./out"


# observation.state 今この瞬間、腕はこの角度にいる。
# action この瞬間、人間の操作では次にこの角度へ動かそうとしている。

def episodeName():
    return f"episode_{EPISODE_INDEX:06d}"


def parquetPath():
    return fr'{DATASET_ROOT}\data\chunk-000\{episodeName()}.parquet'


def videoPathLaptop():
    return\
        fr'{DATASET_ROOT}\videos\chunk-000\observation.images.laptop\{episodeName()}.mp4'


def VideoPathPhone():
    return\
        fr'{DATASET_ROOT}\videos\chunk-000\observation.images.phone\{episodeName()}.mp4'


def saveImageByVideoFrame(frame, videoPath, cameraName):
    print(f" === Save Image by frame {frame} frame ===")
    print(f"Video Path = {videoPath}")
    vid = imageio.get_reader(videoPath, 'ffmpeg')
    img = vid.get_data(frame)
    vid.close()
    Path(VIDEO_OUTPUT_PATH).mkdir(exist_ok=True)
    fn = Path(VIDEO_OUTPUT_PATH) / f'{episodeName()}_frame_{FRAME_INDEX:06d}_{cameraName}.png'

    imageio.imwrite(fn, img)
    print(" === Save Image Done ===")


def getParquetDelta(frame):
    print(" === Delta Action - Observation State === ")
    df = pd.read_parquet(parquetPath())
    row = df.iloc[frame]
    print(f" - frame_index = {row['frame_index']} frame")
    print(f" - timestamp = {row['timestamp']} seconds")
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
        delta = row['action'][n] - row['observation.state'][n]
        print(f"  - {i}: {delta}")


def getParquetFrameAction(frame):
    print("=== Action ===")
    df = pd.read_parquet(parquetPath())
    row = df.iloc[frame]
    print(f" - frame_index = {row['frame_index']} frame")
    print(f" - timestamp = {row['timestamp']} seconds")
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
        print(f"  - {i}: {row['action'][n]}")


def getParquetFrameObsState(frame):
    print("=== Observation State ===")
    df = pd.read_parquet(parquetPath())
    row = df.iloc[frame]
    print(f" - frame_index = {row['frame_index']} frame")
    print(f" - timestamp = {row['timestamp']} seconds")
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
        print(f"  - {i}: {row['observation.state'][n]}")


def main():
    getParquetFrameObsState(FRAME_INDEX)
    getParquetFrameAction(FRAME_INDEX)
    getParquetDelta(FRAME_INDEX)
    saveImageByVideoFrame(FRAME_INDEX, videoPathLaptop(), "laptop")
    saveImageByVideoFrame(FRAME_INDEX, VideoPathPhone(), "phone")


if __name__ == "__main__":
    main()
