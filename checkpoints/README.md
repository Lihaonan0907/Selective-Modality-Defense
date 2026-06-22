# Checkpoints

Model weights are intentionally not tracked in Git.

Prepare the following files locally and point to them in `configs/paths.yaml`:

- four-channel anomaly proposal detector
- DRF-MA learned fusion checkpoint
- visible restoration UNet checkpoint
- infrared restoration UNet checkpoint
- visible task-driven detector
- infrared task-driven detector
- Stable Diffusion inpainting base model
- visible single-branch proposal detector
- infrared single-branch proposal detector
- visible single-branch attribution head
- infrared single-branch attribution head
- optional DRF-MA prototype bank JSON

For public release, provide download links or a script that verifies checksums. See `docs/CHECKPOINTS.md` for the suggested release manifest.
