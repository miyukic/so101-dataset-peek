# so101-dataset-peek: Codex Operating Notes

This repository is part of LocusArm Phase 0.

The user is not learning beginner Python here. Do not explain syntax such as
`print()`, imports, functions, or basic pandas unless explicitly asked.

Default role:

- Explain what the dataset means for SO-101 / LeRobot / ACT / robot learning.
- Treat code, README, git, paths, image/video IO, and formatting as Codex chores.
- Before any new task, state:
  - roadmap position
  - what the human should judge
  - what Codex should handle as chores

For this project, the core learning object is not Python. It is the robot
learning map:

- `observation.images.*`: what the follower arm/cameras saw
- `observation.state`: the follower arm's current joint state
- `action`: the teleoperation target/action recorded from the human demo
- future action chunks: the short motion plan that ACT learns to predict

Do not add throwaway exploration scripts as user-facing steps. If exploration is
needed, Codex may inspect files or run short checks, then explain only the robot
meaning and the next useful result.
