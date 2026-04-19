# SynthExperimentZoo

Workspace and development area for computer vision (CV) experiments using synthetic data for deep learning. The repo contains training/testing pipelines for:

- Single‑image super‑resolution (SR)
- Image‑to‑image translation (paired A→B)
- Semantic segmentation (synthetic → real and variants)

It uses YAML‑driven configurations, task‑agnostic trainers/testers, optional live dashboards via Visdom, and machine presets for different workstations/clusters.

## Stack and Requirements

- Language: Python 3.x (GPU strongly recommended)
- Deep Learning: PyTorch, TorchVision
- Computer Vision: OpenCV, Kornia
- Utilities: NumPy, TQDM, PyYAML, Matplotlib
- Dashboards: Visdom
- Downloads/IO (optional utilities): gdown, requests
- Scheduler (optional): SLURM (see `script_*.slurm` and `slurm_*.sh`)

## Simplified VCC Scheme (Consolidated YAML)
The repository uses a single‑YAML per problem scheme for easier maintenance.

**Format: `problem_version.hyper.loss`**
- **problem**: Selects the YAML file in `configs/` (e.g., `mobisr` -> `configs/mobisr.yaml`).
- **version**: Key in the `versions:` block defining architecture and data paths.
- **hyper**: Index in the `hyperparams:` block for learning rates, etc.
- **loss**: Index in the `loss_weights:` block for weighting functions.

**Example:** `mobisr_v02.06_div2k.12.3`
1. Loads `configs/mobisr.yaml`.
2. Uses architecture settings for `v02.06_div2k`.
3. Sets learning rates from hyperparams entry `12`.
4. Sets loss weights from entry `3`.

**Server Path Injection:**
The YAML contains a `server_paths:` block mapping `--server_config` indices to root directories. Use `{server_path}` in your dataset paths to make them machine-agnostic.

**Usage:**
```bash
python train_sr_main.py --network_version "mobisr_v02.06_div2k.12.3" --server_config 3
```
*Note: `train_sr_main.py` automatically detects if a version string follows this new format and uses the consolidated parser.*

## Configuration Model

### Legacy Layout (`hyperparam_tables/`)
- A version-specific YAML (e.g., `hyperparam_tables/mobisr_v02.05_div2k.yaml`).
- Common hyperparams (`common_hyper.yaml`).
- Common loss-weights (`common_weights.yaml`).

### VCC Layout (`configs/`)
- `configs/models/`: vXX.yaml files defining model architecture.
- `configs/datasets/`: YY.yaml files defining data paths and modality.
- `configs/experiments/`: ZZ.yaml files defining loss weights and learning rates.
- `configs/servers/`: hardware-specific overrides.

## Model IDs
The `ModelFactory` supports the following architecture IDs (used in `model.type` in YAML):

| ID | Model Architecture | ID | Model Architecture |
| :--- | :--- | :--- | :--- |
| **1** | CycleGAN Generator | **8** | DenseNet Generator |
| **2** | UNet Generator | **9** | NAFSSR |
| **3** | AdaIN Generator | **10** | RestormerUNet |
| **4** | FFANet | **11** | PSPNet |
| **5** | RRDBNet | **12** | TranslatorGAN |
| **6** | SwinIR | **13** | StyleTransferGAN |
| **7** | AttentionResUNet | **14** | EmbeddingNetwork |

## Datasets and Directory Layouts

Data roots are selected by `--server_config` (legacy) or `--server` YAML (VCC).

- **Modality: Image**: Standard paired A→B translation.
- **Modality: Video**: Standardized input shape `[B, T, C, H, W]` for temporal processing.

## Entry Points

- `train_vcc_main.py`: Preferred for new experiments (VCC scheme).
- `train_sr_main.py`: Super-resolution (supports both schemes).
- `train_img2img_main.py`: General image-to-image.
- `train_seg_main.py`: Semantic segmentation.

## Project Structure

- `core/`: Factory classes (Model, Dataset, Loss) and ScalableTrainer.
- `configs/`: Hierarchical YAML configurations for the VCC scheme.
- `utils/config_parser.py`: VCC string parser and YAML merger.
- `model/`: Neural network architectures.
- `trainers/`: trainer implementations.
- `loaders/`: data loaders.

## License
This project is licensed under the MIT License. See `LICENSE`.
