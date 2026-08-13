# DrivoR HUGSIM Client

This repo (fork of [valeoai/DrivoR](https://github.com/valeoai/DrivoR)) is the
implementation of a DrivoR test client for the HUGSIM closed-loop benchmark,
alongside the same benchmark's UniAD/VAD/LTF clients.

The implementation is based on:

[Driving on Registers](https://valeoai.github.io/driving-on-registers/)

[arXiv Paper](https://arxiv.org/abs/2601.05083), CVPR 2026

## Installation

Please refer to this repo's own [Installation](README.md#installations)
instructions. In practice this fork's environment was migrated from
`conda` to `uv` (env at `uv-envs/drivoR`, `torch==2.1.0+cu121`); adjust
`drivor_e2e.sh`'s interpreter path if your setup differs.

Please change `DRIVOR_PATH` in `drivor_e2e.sh` (and `_checkpoint_path`/
`_config_path` at the top of `hugsim_drivor_client_raw.py`) to the path on
your machine.

## Launch Client

### Manually Launch

```bash
zsh ./drivor_e2e.sh ${CUDA_ID} ${output_dir}
```

### Auto Launch

The client can be auto-launched by HUGSIM's closed-loop script --
specifically `closed_loop_drivor.py --ad drivor` (not `closed_loop.py`,
which only wires up UniAD/VAD/LTF), reading `drivor_path` from
`configs/sim/<dataset>_base.yaml` in the HUGSIM repo.

## Debug images

`hugsim_drivor_client_raw.py` writes one debug frame per camera per
simulation step under `${output_dir}/drivor/debug_imgs/`, split into three
stages so a bad frame can be traced back to where it went wrong:

| folder | what it captures |
|---|---|
| `1_hugsim_raw/` | The raw RGB frame exactly as received from HUGSIM over `obs_pipe`, before any resizing/processing. Use this to rule out HUGSIM-side rendering issues. |
| `2_precam/` | The same frame right before it's wrapped into a navsim `Camera` object (`create_cam`). Currently identical to `1_hugsim_raw/` (no resize applied), but stays a separate stage since that's where a resize would go back in if reintroduced. |
| `3_feature/` | The per-camera feature map (`features['image']`) DrivoR's feature builder produces, immediately before it's fed to the model -- min-max normalized to 0-255 for viewing. Use this to catch feature-extraction bugs (wrong channel order, dead/garbled features) independent of the raw input. |

Filenames are `{frame_idx:04d}_{cam_name}.jpg` within each folder.
