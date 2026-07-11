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

`show_batch.py` は、指定したepisode/frameについて、parquetの `observation.state` / `action` と、同じframeのlaptop/phone動画画像を対応させる。

`check_act_shapes.py` は、ACTに渡す前提として、画像・現在関節状態・actionのshapeを表示する。現時点では1フレーム単位のshape確認で、future action chunk `[50, 6]` の生成までは含めていない。

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

このデータセットは、SO-101のデモ操作ログとして読める。

SO-101では、人間が手で動かす `leader arm` と、作業空間で実際に動く `follower arm` の2本を使ってデモを記録する。人間はleaderを動かし、followerがその操作に追従する。その間に、カメラ映像と関節角が同じ時系列で保存される。

1行は、1フレーム相当の記録。

```text
observation.images.* = follower arm が作業している映像
observation.state    = follower arm のその瞬間の実際の関節角
action               = leader arm 由来の人間操作目標
```

`action` は「次フレームの `observation.state` そのもの」ではない。サーボの追従遅れ、負荷、速度制限、キャリブレーション差などがあるため、followerは目標値へ向かうが、実際の姿勢は完全には一致しないことがある。

ACT目線では、あとで以下の対応を学習する入口になる。

```text
camera image + observation.state -> future action chunk
```

つまり、この段階では「モデルを作る」前に、ACTが学習する材料が画像・現在姿勢・人間の操作目標として読めるかを確認している。

## Setup

```powershell
git clone https://github.com/miyukic/so101-dataset-peek.git
cd so101-dataset-peek
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

データセット本体はGitHubには含めない。ローカルでは以下に置く想定。

```text
.\data\orsoromeo_so101_pick_and_place
```

取得例:

```powershell
.\.venv\Scripts\huggingface-cli.exe download orsoromeo/so101_pick_and_place --repo-type dataset --local-dir .\data\orsoromeo_so101_pick_and_place
```

## Run

```powershell
.\.venv\Scripts\python.exe .\peek_dataset.py
```

任意の episode / frame と動画フレームの対応を見る。

```powershell
.\.venv\Scripts\python.exe .\show_batch.py
```

`show_batch.py` は `EPISODE_INDEX` と `FRAME_INDEX` を変更して使う。実行すると、指定frameの `observation.state` / `action` / 差分を表示し、同じframeの laptop / phone 画像を `out/` に保存する。

ACTへ渡す前提のshapeを見る。

```powershell
.\.venv\Scripts\python.exe .\check_act_shapes.py
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

- 第0部の理解をZenn記事として整理する
- future action chunk `[50, 6]` を可視化するか判断する
- SO-101購入前に、印刷・電源・サーボ1個テストのゲートを確認する
