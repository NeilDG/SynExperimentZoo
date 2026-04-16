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

Notes
- CUDA/cuDNN is required for GPU training. Install the torch build matching your CUDA.
- No `requirements.txt` or Conda env file is committed yet.
  - TODO: Pin exact Python, CUDA, and dependency versions; add `environment.yml` and/or `requirements.txt`.

### Example setup (Conda, then pip)

```
conda create -n synzoo python=3.10  # TODO: confirm supported Python version(s)
conda activate synzoo

# Install PyTorch matching your CUDA (example below is a placeholder)
# TODO: replace with the correct command from https://pytorch.org/get-started/locally/
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Core dependencies
pip install opencv-python kornia numpy tqdm pyyaml matplotlib visdom gdown requests

# Start Visdom (optional, used when plotting is enabled)
python -m visdom.server --port 8097
```


## Configuration Model

Configurations are YAML‑driven and loaded by `config/network_config.py` via the `ConfigHolder` singleton. Three inputs are used per run:

1) A version‑specific YAML that defines the model/dataset wiring (e.g., `hyperparam_tables/mobisr_v02.05_div2k.yaml`).
2) A common hyperparameters table (`hyperparam_tables/*/common_hyper.yaml` or `hyperparam_tables/common_hyper.yaml`).
3) A common loss‑weights table (`hyperparam_tables/*/common_weights.yaml` or `hyperparam_tables/common_weights.yaml`).

The CLI flag `--network_version` follows a 4‑part dotted format parsed by `util_script_main.parse_string`:

- Format: `AA.BB.CC.DD`
  - `AA.BB` → selects the version‑specific YAML file named `AA.BB.yaml` under the appropriate `hyperparam_tables` folder
  - `CC` → selects the hyperparameters entry index in `common_hyper.yaml`
  - `DD` → selects the loss‑weights entry index in `common_weights.yaml`

Examples
- `mobisr_v02.05_div2k.10.1` → loads `hyperparam_tables/mobisr_v02.05_div2k.yaml` with hyperparams index `10` and loss‑weights index `1`.
- `fcg2cityscapes_v00.00.05.4` → loads `hyperparam_tables/img2img/fcg2cityscapes_v00.00.yaml` with hyperparams index `5` and loss‑weights index `4`.

Global runtime toggles and data paths are in `global_config.py`, but are normally set by each entry script’s `update_config()` function based on `--server_config` presets.


## Datasets and Directory Layouts

Data roots are selected by `--server_config` (machine preset), and specific file globs are built from the loaded YAML. Typical patterns by task:

- Super‑Resolution (SR)
  - YAML keys: `dataset_version`, `low_path`, `high_path`
  - Train/Test paths (examples; vary by preset):
    - `.../SuperRes Dataset/{dataset_version}/low/train_patches/*.jpg`
    - `.../SuperRes Dataset/{dataset_version}/high/train_patches/*.jpg`
    - `.../SuperRes Dataset/{dataset_version}/low/test_images/*.jpg`
    - `.../SuperRes Dataset/{dataset_version}/high/test_images/*.jpg`
  - Some training runs append `_patched` to `dataset_version` (see `train_sr_main.py` TODO note in code).

- Image‑to‑Image (A→B)
  - YAML keys: `dataset_a_train`, `dataset_b_train`, `dataset_a_test`, `dataset_b_test`
  - Resolved patterns (examples; vary by preset):
    - `.../Datasets/{dataset_version}`

- Segmentation
  - YAML keys: `dataset_version`, `img_path_train`, `mask_path_train`, `img_path_test`, `mask_path_test`
  - Resolved patterns (examples; vary by preset):
    - `.../Segmentation Dataset/{dataset_version}/{img_path}`
    - `.../Segmentation Dataset/{dataset_version}/{mask_path}`
  - Some runs also use `_patched` in `dataset_version`.

Utilities
- Patchification helpers in `util_script_main.py` for creating `*_patched` datasets using Kornia’s `extract_tensor_patches`.
- Hypersim downloader in `utils/ml_hypersim_dl.py` (uses `requests`).
- Google Drive downloads via `gdown_download.py`.

TODOs
- Document dataset acquisition steps for Div2K/Flickr2K/BurstSR/Cityscapes/FCG/etc., including licenses/links.
- Provide small sample data or scripts to generate minimal toy datasets for quick verification.


## Entry Points and How to Run

All main entry scripts expose a small set of common flags (exact set can vary per script):
- `--server_config` (int): selects a machine preset (paths, batch sizes, workers, and plotting defaults)
- `--cuda_device` (str): device string, e.g., `cuda:0`
- `--img_to_load` (int): optional index filter for loading
- `--network_version` (str): version string selecting YAML + hyper/loss indices
- `--plot_enabled` (int): `1` to enable Visdom figures; `0` to disable
- `--save_per_iter` (int): snapshot/plot frequency (where applicable)
- `--save_images` (int): tester option to dump output images

Start a Visdom server if you set `--plot_enabled=1`:
```
python -m visdom.server --port 8097
```
Special case: if `global_config.server_config == -99`, Visdom will try to connect to `192.168.134.223:8097` (see `utils/plot_utils.py`).

### Super‑Resolution (SR)
- Train
```
python train_sr_main.py \
  --server_config=0 \
  --cuda_device="cuda:0" \
  --plot_enabled=1 \
  --save_per_iter=500 \
  --network_version="mobisr_v02.05_div2k.10.1"
```
- Test
```
python test_sr_main.py \
  --server_config=0 \
  --cuda_device="cuda:0" \
  --plot_enabled=0 \
  --save_images=1 \
  --network_version="mobisr_v02.05_div2k.10.1"
```

### Image‑to‑Image
- Train
```
python train_img2img_main.py \
  --server_config=0 \
  --cuda_device="cuda:0" \
  --plot_enabled=1 \
  --save_per_iter=500 \
  --network_version="fcg2cityscapes_v00.00.05.4"
```

### Segmentation
- Train
```
python train_seg_main.py \
  --server_config=0 \
  --cuda_device="cuda:0" \
  --plot_enabled=1 \
  --save_per_iter=500 \
  --network_version="synseg_v00.00_cityscapes.01.1"
```
- Alternative trainer (same flags; dataset wiring differs in code): `train_seg_main_2.py`

### Hardware‑specific wrappers
There are convenience wrappers that chain multiple runs for a specific machine, e.g. `g411_4060ti-pc1_main.py`, `titan*_3060_main.py`, `g411_5090-pc*.py`, etc. Inspect and edit the `main()` function in these files to enable the runs you want, then execute:
```
python g411_4060ti-pc1_main.py
```

### SLURM scripts (optional)
SLURM helper files are provided for HPC environments:
- `script_*.slurm`, `script_a100_*.slurm`, `script_debug.slurm`, `script_util.slurm`
- Setup/uninstall helpers: `slurm_install.sh`, `slurm_install_cuda118.sh`, `slurm_uninstall.sh`
- Downloads: `slurm_download.sh`, `gdown_download.py`

Usage varies by cluster. Typical pattern:
```
sbatch script_1.slurm
```
TODO: Document SLURM partition names, GPUs, modules, and environment specifics per cluster (e.g., CCS/COARE) and map them to `--server_config`.


## Server Presets (`--server_config`)
Each entry script defines its own mapping of `server_config` integers to preset paths/batch sizes/worker counts. Common values seen in code include:
- `0`: Local Windows workstation (e.g., RTX 4060 Ti) using `C:/Datasets/...`
- `1`: CCS Cloud (Linux paths under `/home/...`); sometimes disables progress bars and plots
- `2`: RTX 2080 Ti machine (Windows `X:/...`)
- `3`: RTX 3090 machine (Windows `X:/...`)
- `4`: TITAN workstation variants (Linux `~/Documents/...`)
- `5`: High‑end workstation variant (e.g., RTX 5090 in SR scripts) or TITAN RTX 2070 in other scripts
- `6`: G411 RTX 3060 workstation (Windows `C:/...`)
- `7`: Laguna RTX 3060 PCs (Windows `D:/...`)
- `8`: DOST‑COARE cluster (Linux `/scratch3/...`)

Important
- This mapping differs slightly across scripts (SR, img2img, seg). Check the `update_config()` function in the specific script you are running for the exact paths and batch sizes.


## Tests
There is no unit‑test framework configured. Instead, task‑level tester scripts are provided:
- Super‑Resolution: `test_sr_main.py`
- Image‑to‑Image: `test_img2img_main.py`
- Segmentation: `test_seg_main.py`

Example
```
python test_img2img_main.py --server_config=3 --plot_enabled=1 --network_version="synth2srd_v01.00.5.1"
```

TODO: Add automated tests (e.g., `pytest`) and small fixtures to validate loaders and trainers.


## Project Structure (selected)

- Root training/testing scripts: `train_sr_main.py`, `test_sr_main.py`, `train_img2img_main.py`, `test_img2img_main.py`, `train_seg_main.py`, `train_seg_main_2.py`, `test_seg_main.py`
- Config and tables:
  - `config/network_config.py` – runtime config holder
  - `hyperparam_tables/` – version YAMLs and common hyper/loss tables
- Data loading:
  - `loaders/dataset_loader.py` – central factory for datasets
  - `loaders/superres_datasets.py`, `loaders/segmentation_datasets.py`
- Models and training:
  - `model/`, `model/modules/` – architectures and components
  - `trainers/` – trainers (paired, img2img, segmentation, template)
  - `testers/` – evaluation logic
- Utilities and transforms:
  - `utils/` – plotting (Visdom), conversions, dataset tools, Hypersim downloader
  - `transforms/`, `processing/` – data transforms and dataset creation utilities
- Reports and outputs: `reports/` – sample outputs and logs
- Schedulers/scripts: `script_*.slurm`, `slurm_*.sh`, `ccs_gpu_*.sh`, `visdom_run.slurm`


## Environment variables and external services
- Environment variables: none required by default (paths are configured by presets and YAML values).
- External services: Visdom dashboard (`python -m visdom.server --port 8097`).
  - Special preset `server_config = -99` targets `192.168.134.223:8097`.


## License
This project is licensed under the MIT License. See `LICENSE`.


## Changelog / Roadmap
- TODO: Publish a `requirements.txt` / `environment.yml` with pinned versions and CUDA matrix.
- TODO: Add instructions and links for downloading/preparing datasets (Div2K, Flickr2K, BurstSR, Cityscapes, FCG, Hypersim, etc.).
- TODO: Parameterize dataset roots to avoid hard‑coded paths in `update_config()`.
- TODO: Provide sample configs and a quickstart script that runs on CPU with toy data.
