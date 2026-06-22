# Selective Modality Defense

Research code for **Selective Modality Defense: Attributing and Restoring Adversarial Patches in Dual-Modal Vision**.

Selective Modality Defense targets visible-infrared perception systems where adversarial patches may corrupt the visible branch, the infrared branch, or both. The pipeline proposes suspicious regions, attributes each proposal to the corrupted modality, restores only the affected branch, and runs task-driven single-modal detectors on the restored outputs.

## What Is Included

* Four-channel visible/infrared anomaly proposal code.

* Self-contained DRF-MA evidence extraction, learned fusion training, evaluation, and ablation scripts.

* Attribution-guided asymmetric mask generation.

* Visible and infrared diffusion-restoration wrappers.

* Task-driven detector wrappers and end-to-end paired/single-branch inference.

* Single-branch visible-only and infrared-only proposal, attribution, restoration, and evaluation entry points.

* Release-oriented configuration templates and checkpoint documentation.

Datasets, model weights, generated outputs, and machine-local paths are intentionally not tracked.

## Repository Layout

```text
configs/       YAML configs and local path template
docs/          method, config, checkpoint, and release notes
scripts/       command-line entry points
src/
  proposal/    four-channel proposal and anomaly utilities
  attribution/ DRF-MA evidence, learned fusion, metrics, masks
  restoration/ modality-specific inpainting restorers
  detection/   YOLO detector adapters
  pipeline/    paired and single-branch inference
  utils/       config, image IO, logging, and shared helpers
```

## Installation

```bash
conda env create -f environment.yml
conda activate selective-modality-defense
pip install -r requirements.txt
```

If you install with plain `pip`, use Python 3.10 or newer and install the PyTorch build that matches your CUDA/CPU environment.

## Local Paths

Copy the path template and fill in local dataset, checkpoint, and model locations:

```bash
cp configs/paths.yaml.example configs/paths.yaml
```

`configs/paths.yaml` is ignored by Git. The public repository should not contain private absolute paths.

Required for end-to-end inference:

* `checkpoints.proposal`

* `checkpoints.drf_ma`

* `checkpoints.vis_restorer`

* `checkpoints.ir_restorer`

* `checkpoints.vis_detector`

* `checkpoints.ir_detector`

* `models.stable_diffusion_inpaint`

Required for single-branch inference:

* `checkpoints.single_branch.vis_proposal`

* `checkpoints.single_branch.ir_proposal`

* `checkpoints.single_branch.vis_attribution`

* `checkpoints.single_branch.ir_attribution`

`paths.legacy_code_root` is optional. It is only needed by the legacy compatibility wrappers for four-channel proposal, task-detector training, or paired DRF-MA adapter experiments.

## Quick Start

Paired visible-infrared inference:

```bash
python scripts/run_defense.py \
  --config configs/default.yaml \
  --visible path/to/visible.png \
  --infrared path/to/infrared.png \
  --output outputs/demo_pair
```

Single-branch infrared inference:

```bash
python scripts/run_defense.py \
  --config configs/default.yaml \
  --mode ir \
  --image path/to/infrared.png \
  --output outputs/demo_ir
```

Outputs include restored images, restoration masks, detections, attribution records, and `summary.json`.

## DRF-MA Training And Evaluation

The DRF-MA learned attribution path is self-contained in this repository. Set `paths.data_root` in `configs/paths.yaml`, then run:

```bash
python scripts/train_drf_ma.py --config configs/drf_ma.yaml
python scripts/eval_drf_ma.py --config configs/drf_ma.yaml --checkpoint outputs/drf_ma/best.pt
python scripts/ablate_drf_ma.py --config configs/drf_ma.yaml
```

For a quick syntax and data-layout smoke test, use `--max-samples`, `--epochs`, and `--batch-size`:

```bash
python scripts/train_drf_ma.py \
  --config configs/drf_ma.yaml \
  --max-samples 8 \
  --epochs 1 \
  --batch-size 4
```

For formal validation, DRF-MA proposal fallback from true patch boxes is disabled outside the configured oracle fallback splits. Keep `proposal.allow_oracle_fallback_splits: [train]` unless you are explicitly running an oracle/debug experiment.

## Single-Branch Variant

The single-branch variant disables cross-modal discrepancy, branch selection, and dominance correction. It uses only the available modality:

```bash
python scripts/infer_single_branch_proposal.py \
  --config configs/single_branch_ir.yaml \
  --mode ir \
  --image path/to/infrared.png \
  --output outputs/ir_proposals

python scripts/train_single_branch_attribution.py \
  --config configs/single_branch_ir.yaml \
  --mode ir

python scripts/eval_single_branch_detection.py \
  --config configs/single_branch_ir.yaml \
  --mode ir \
  --data-root path/to/single_branch_dataset \
  --output outputs/ir_eval
```

Visible mode uses `configs/single_branch_vis.yaml` and `--mode vis`. During inference and detection evaluation, true patch masks are not used to generate proposals or restoration masks.

## Legacy Training Wrappers

Proposal and detector training wrappers can still forward to the original research training scripts through `paths.legacy_code_root`:

```bash
python scripts/train_proposal.py --config configs/default.yaml
python scripts/train_detector.py --config configs/default.yaml
```

Restoration fine-tuning is self-contained in this repository:

```bash
python scripts/train_restoration.py --config configs/default.yaml --modality vis
python scripts/train_restoration.py --config configs/default.yaml --modality ir
```

## Acknowledgements

This project builds on PyTorch, Ultralytics YOLO, Hugging Face diffusers, and visible-infrared pedestrian perception datasets and benchmarks.
