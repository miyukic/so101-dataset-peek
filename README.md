# so101-dataset-peek

SO-101 / LeRobot dataset の中身を確認するための小さな実験リポジトリ。

対象データセット:

- Hugging Face: `orsoromeo/so101_pick_and_place`
- robot type: `so101`
- episodes: `50`
- frames: `22235`
- fps: `30`
- cameras: `observation.images.laptop`, `observation.images.phone`

このリポジトリの第0部では、ACTや模倣学習へ進む前に、データセットの1行が何を表しているかを読む。

## What This Checks

`peek_dataset.py` は以下を表示する。

- `meta/info.json` の概要
- 先頭 parquet の shape / columns / dtype
- 先頭行の `action` と `observation.state`
- `action` と `observation.state` がどちらも6次元であること
- 6次元がSO-101の関節名に対応すること

確認した関節名:

```text
main_shoulder_pan
main_shoulder_lift
main_elbow_flex
main_wrist_flex
main_wrist_roll
main_gripper
```

## Meaning

1行は、1フレーム相当のロボット操作ログとして読める。

```text
observation.state = その瞬間のロボットの現在姿勢
action            = その瞬間、人間操作では次に向かわせた関節値
```

ACT目線では、あとで以下の対応を学習する入口になる。

```text
camera image + observation.state -> action
```

つまり、この段階では「モデルを作る」前に、学習データの材料が画像・現在姿勢・操作指令として読めるかを確認している。

## Setup

```powershell
cd F:\source1\so101-dataset-peek
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

データセット本体はGitHubには含めない。ローカルでは以下に置く想定。

```text
F:\source1\so101-dataset-peek\data\orsoromeo_so101_pick_and_place
```

取得例:

```powershell
.\.venv\Scripts\huggingface-cli.exe download orsoromeo/so101_pick_and_place --repo-type dataset --local-dir .\data\orsoromeo_so101_pick_and_place
```

## Run

```powershell
cd F:\source1\so101-dataset-peek
.\.venv\Scripts\python.exe .\peek_dataset.py
```

## Example Output

```text
=== Dataset Overview ===
  robot_type: so101
  total_episodes: 50
  total_frames: 22235
  fps: 30

=== First Parquet ===
  shape: (447, 7)

=== actionとobservation.state の 次元数 ===
  - action次元: 6
  - observation.state次元: 6

=== First Row Observation State ===
  - main_shoulder_pan: -27.24609375
  - main_shoulder_lift: 194.326171875
  - main_elbow_flex: 183.955078125
  - main_wrist_flex: 61.611328125
  - main_wrist_roll: 100.1953125
  - main_gripper: 11.610793113708496

=== First Row Action ===
  - main_shoulder_pan: -26.982421875
  - main_shoulder_lift: 196.69921875
  - main_elbow_flex: 182.724609375
  - main_wrist_flex: 61.611328125
  - main_wrist_roll: 99.931640625
  - main_gripper: 11.65803050994873
```

## Next

- 任意 episode / frame を選んで表示する
- `observation.images.laptop` / `observation.images.phone` の動画と parquet のフレーム対応を見る
- ACT入力へ渡す前提の shape 表を作る
