# DrivoR HUGSIM Client

[한국어](README_HUGSIM.ko.md)

---

This repo (fork of [valeoai/DrivoR](https://github.com/valeoai/DrivoR)) is the
implementation of a DrivoR test client for the
[HUGSIM](https://github.com/hyzhou404/HUGSIM) closed-loop benchmark,
alongside the same benchmark's UniAD/VAD/LTF clients.

The implementation is based on:

[Driving on Registers](https://valeoai.github.io/driving-on-registers/)

[arXiv Paper](https://arxiv.org/abs/2601.05083), CVPR 2026

---

## 1. Changes vs. upstream `valeoai/DrivoR`

This isn't a plain fork -- the following was added on top of upstream
specifically for HUGSIM integration (none of it exists in
[valeoai/DrivoR](https://github.com/valeoai/DrivoR)):

- **`drivor_e2e.sh`** -- the launcher script `sim/utils/launch_ad.py::launch()`
  calls on the HUGSIM side (`zsh ${shell_path} ${cuda_id} ${output}`); takes
  `(cuda_id, output_dir)` as positional args and execs the client entry point
  below.
- **`hugsim_drivor_client_raw.py`** -- the actual entry point (confirmed:
  `drivor_e2e.sh` execs this file, not the similarly-named
  `hugsim_drivor_client.py` also present in this repo -- what that other file
  is for isn't documented here). Talks to HUGSIM over the `obs_pipe`/
  `plan_pipe` named pipes (§5), converts observations into DrivoR's model
  input format, runs inference, writes the `(N, 2)` trajectory back.
- On the **HUGSIM side** (separate repo), `closed_loop_drivor.py` was added
  alongside `closed_loop.py` -- `closed_loop.py` itself only wires up
  UniAD/VAD/LTF and wasn't reused since this client's obs/info shape and
  post-loop save/eval step differ.

For everything else (base DrivoR/navsim model, training, upstream evaluation
scripts like `eval.sh`/`metric_caching.sh`), see this repo's own
[README.md](README.md) -- not duplicated here.

---

## 2. Installation

Please refer to this repo's own [Installation](README.md#installations)
instructions for the base DrivoR/navsim environment. In practice this fork's
environment was migrated from `conda` to `uv` (`torch==2.1.0+cu121`);
`drivor_e2e.sh` assumes that but reads the actual interpreter path from an
env var, so any working DrivoR env works.

**경로 설정 (env var override)**: `drivor_e2e.sh`의 `DRIVOR_PATH`/
`DRIVOR_PYTHON`, `hugsim_drivor_client_raw.py`의 `DRIVOR_CHECKPOINT_PATH`/
`DRIVOR_CONFIG_PATH`는 전부 env var로 override 가능 (기본값은 placeholder).
자신의 환경에 맞게 설정할 것:

```bash
export DRIVOR_PATH=/path/to/your/DrivoR
export DRIVOR_PYTHON=/path/to/your/DrivoR/venv/bin/python
export DRIVOR_CHECKPOINT_PATH=/path/to/your/checkpoint.pth
export DRIVOR_CONFIG_PATH=/path/to/your/DrivoR/navsim/planning/script/config/common/agent/drivoR.yaml
```

---

## 3. Launch Client

### Manually Launch

```bash
zsh ./drivor_e2e.sh ${CUDA_ID} ${output_dir}
```

### Auto Launch (closed-loop evaluation)

In practice you don't call `drivor_e2e.sh` directly -- HUGSIM's closed-loop
script launches it per scenario. Run this from the **HUGSIM repo** (inside
its `hugsim_v3` docker environment, not this repo's env):

```bash
sim_cuda=0
ad_cuda=1

# your scenario yaml directory on the HUGSIM side
scenario_dir=${SCENARIO_PATH}

for cfg in ${scenario_dir}/*.yaml; do
    echo ${cfg}
    CUDA_VISIBLE_DEVICES=${sim_cuda} \
    python closed_loop_drivor.py --scenario_path ${cfg} \
                        --base_path ./configs/sim/${dataset_name}_base.yaml \
                        --camera_path ./configs/sim/${dataset_name}_camera.yaml \
                        --kinematic_path ./configs/sim/kinematic.yaml \
                        --ad drivor \
                        --ad_cuda ${ad_cuda}
done
```

This is `closed_loop_drivor.py --ad drivor` (**not** `closed_loop.py`, which
only wires up UniAD/VAD/LTF). It reads `drivor_path` from
`configs/sim/<dataset>_base.yaml` -- point that at this repo's checkout (the
same path you set as `DRIVOR_PATH` in §2 above), and it execs `drivor_e2e.sh`
for you per scenario, feeding it through the `obs_pipe`/`plan_pipe` protocol
in §5.

---

## 4. Debug images

`hugsim_drivor_client_raw.py` writes one debug frame per camera per
simulation step under `${output_dir}/drivor/debug_imgs/`, split into three
stages so a bad frame can be traced back to where it went wrong:

| folder | what it captures |
|---|---|
| `1_hugsim_raw/` | The raw RGB frame exactly as received from HUGSIM over `obs_pipe`, before any resizing/processing. Use this to rule out HUGSIM-side rendering issues. |
| `2_precam/` | The same frame right before it's wrapped into a navsim `Camera` object (`create_cam`). Currently identical to `1_hugsim_raw/` (no resize applied), but stays a separate stage since that's where a resize would go back in if reintroduced. |
| `3_feature/` | The per-camera feature map (`features['image']`) DrivoR's feature builder produces, immediately before it's fed to the model -- min-max normalized to 0-255 for viewing. Use this to catch feature-extraction bugs (wrong channel order, dead/garbled features) independent of the raw input. |

Filenames are `{frame_idx:04d}_{cam_name}.jpg` within each folder.

---

## 5. Protocol: how HUGSIM and this client talk to each other

Every AD client (this one, UniAD_SIM, VAD_SIM, NAVSIM) is its own standalone
repo/process -- HUGSIM never imports the AD algorithm's code directly. They
communicate over **two OS named pipes (FIFOs)** created by HUGSIM's side in
the run's output directory:

```text
${output_dir}/obs_pipe    -- HUGSIM writes, client reads (observation)
${output_dir}/plan_pipe   -- client writes, HUGSIM reads (planned trajectory)
```

One simulation step looks like this (see `closed_loop_drivor.py::create_gym_env`
on the HUGSIM side, `hugsim_drivor_client_raw.py`'s main loop on this side):

```text
HUGSIM                                    AD client (this repo)
──────────────────────────────────────    ──────────────────────────────────
env.step() -> (obs, info)
pickle.dumps((obs, info))
  -> write to obs_pipe          ────────>  read obs_pipe, pickle.loads()
                                            build model input from obs/info
                                            run model -> trajectory
                                            pickle.dumps(plan_traj)
  read plan_pipe, pickle.loads() <────────   -> write to plan_pipe
traj2control(plan_traj, info)
  -> action -> env.step(action)
(loop back to env.step())
```

At the very end, HUGSIM writes the literal string `'Done'` (not a tuple) to
`obs_pipe` once, instead of `(obs, info)` -- the client must check for this
and exit cleanly (see `hugsim_drivor_client_raw.py`'s `if raw_data == 'Done'`
branch) rather than trying to unpack it as `(obs, info)`.

### `obs` dict

```python
obs = {
    'rgb': {
        'CAM_FRONT': <np.ndarray, HxWx3, RGB>,
        'CAM_FRONT_LEFT': ...,
        'CAM_FRONT_RIGHT': ...,
        'CAM_BACK': ...,
        'CAM_BACK_LEFT': ...,
        'CAM_BACK_RIGHT': ...,
    }
}
```

### `info` dict (the fields this client actually reads)

| key | meaning |
|---|---|
| `command` | discrete navigation command index (e.g. straight/left/right), used to build a one-hot `command` vector |
| `ego_pos` | ego position in HUGSIM's world/OpenCV-style axes -- this client rotates it into IMU axes via `OPENCV2IMU` |
| `ego_steer` | ego heading/yaw |
| `ego_velo`, `accelerate` | scalar forward speed / acceleration, decomposed into x/y via the yaw above |
| `cam_params` | per-camera dict, each with `intrinsic` (`fovx`/`fovy`/`H`/`W`/`cx`/`cy`) and `l2c` (lidar-to-camera 4x4) -- used to build each camera's projection for the model |
| `ego_box`, `obj_boxes`, `collision`, `rc`, `timestamp` | logged by HUGSIM into `data.pkl`, not consumed by this client's model itself |

### `plan_traj` (what the client must write back)

A `numpy.ndarray` of shape `(N, 2)` -- `N` future waypoints in `(x, y)`
ego-relative coordinates (`way_points[:, :2]` in
`hugsim_drivor_client_raw.py`; this client's model actually predicts `(N, 3)`
including a third axis, but only the first two columns go back over the
pipe). HUGSIM's `traj2control()` converts this directly into
acceleration/steer-rate control -- **do not** hand back a fundamentally
different trajectory length/convention without checking `traj2control`'s
assumptions on the HUGSIM side first.

---

## 6. Adding a different AD algorithm

If you want to swap DrivoR for a different model (or add a second one
alongside it), the pattern to follow is exactly what UniAD_SIM/VAD_SIM/
NAVSIM/DrivoR each already do -- **a new standalone repo, not code merged
into HUGSIM or into this repo**:

1. **New repo, own environment.** The AD algorithm keeps its own
   dependencies/checkpoint/config entirely separate from HUGSIM's -- this is
   why DrivoR runs its own `uv` env with a different torch build than
   HUGSIM's own container.

2. **One entry-point script** (this repo's `hugsim_drivor_client_raw.py`,
   `drivor_e2e.sh`) that:
   - creates `obs_pipe`/`plan_pipe` at `${output}/{obs,plan}_pipe` if they
     don't exist yet (`os.mkfifo`)
   - loops: read `obs_pipe` -> `pickle.loads` -> if `'Done'`, clean up and
     exit -> else unpack `(obs, info)` -> build your model's input format
     from the `obs`/`info` fields documented in §5 -> run inference -> pack
     the result as an `(N, 2)` `numpy.ndarray` -> `pickle.dumps` -> write to
     `plan_pipe`

3. **A launcher shell script** (`drivor_e2e.sh`'s role) taking
   `(cuda_id, output_dir)` positionally -- this is the exact signature
   `sim/utils/launch_ad.py::launch()` invokes with
   (`zsh ${shell_path} ${cuda_id} ${output}`). Any interpreter/shell works as
   long as it accepts those two positional args and launches your entry-point
   script correctly.

4. **Wire it into HUGSIM's config**, one of two ways:
   - **Preferred (matches existing clients)**: add a new `--ad <name>` case
     alongside `uniad`/`vad`/`ltf` in `closed_loop.py` (if your client's
     observation needs match what `closed_loop.py` already builds), or add a
     new `closed_loop_<name>.py` mirroring `closed_loop_drivor.py` if your
     client needs a different obs/info shape or a different post-loop
     save/eval step (this is why DrivoR gets its own `closed_loop_drivor.py`
     instead of reusing `closed_loop.py`).
   - Either way, add a `<name>_path` field to
     `configs/sim/<dataset>_base.yaml` pointing at your new repo, following
     the existing `uniad_path`/`vad_path`/`ltf_path`/`drivor_path` fields.

5. **Test the pipe protocol in isolation first** if the model itself isn't
   ready yet -- a stub client that just reads `obs_pipe` and writes back a
   straight-line `(N, 2)` trajectory is enough to confirm the plumbing (pipe
   creation, `'Done'` handling, `traj2control` accepting your trajectory
   shape) before wiring in real inference.

No part of this protocol is DrivoR-specific -- everything in §5 is HUGSIM's
own closed-loop contract, so a from-scratch client can be built by reading
`closed_loop_drivor.py`/`closed_loop.py` on the HUGSIM side and this file's
§5 without needing to understand DrivoR's own model code at all.
