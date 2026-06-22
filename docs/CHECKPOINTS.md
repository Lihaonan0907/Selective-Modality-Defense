# Checkpoints

Model weights are not tracked in this repository. Keep them outside Git and reference them from `configs/paths.yaml`.

## Required For End-To-End Inference

| Component | Config key |
|---|---|
| Four-channel proposal detector | `checkpoints.proposal` |
| DRF-MA learned fusion | `checkpoints.drf_ma` |
| Visible restoration expert | `checkpoints.vis_restorer` |
| Infrared restoration expert | `checkpoints.ir_restorer` |
| Visible task detector | `checkpoints.vis_detector` |
| Infrared task detector | `checkpoints.ir_detector` |
| Stable Diffusion inpainting base model | `models.stable_diffusion_inpaint` |

## Required For Single-Branch Inference

| Component | Config key |
|---|---|
| Visible single-branch proposal detector | `checkpoints.single_branch.vis_proposal` |
| Infrared single-branch proposal detector | `checkpoints.single_branch.ir_proposal` |
| Visible single-branch attribution head | `checkpoints.single_branch.vis_attribution` |
| Infrared single-branch attribution head | `checkpoints.single_branch.ir_attribution` |

## Suggested Release Manifest

For each public checkpoint, provide:

- Download URL.
- Expected filename.
- SHA256 checksum.
- Compatible config file and config key.
- Training dataset/version notes.
- License or redistribution terms.

Example:

```text
proposal.pt
  config: checkpoints.proposal
  sha256: <fill-before-release>
  url: <fill-before-release>
```

## Git Safety

`.gitignore` excludes common PyTorch, YOLO, ONNX, TensorRT, NumPy, and pickle artifacts. Before publishing, still run a manual check for accidentally tracked weights or generated result files.
