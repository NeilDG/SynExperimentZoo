# Visualization Modernization Plan

**Project:** SynExperimentZoo - Super Resolution Phase (`train_sr_main.py`)  
**Objective:** Transition from legacy Visdom/Matplotlib reporting to a modern, scalable, and SR-specific visualization suite.

---

## Phase 1: Performance & Memory Optimization (The "Clean-Up")
*Goal: Remove CPU bottlenecks and RAM leaks in the current training loop.*

1.  **Metric Buffer Refactor (Missing Circular Buffers):**
    *   **Problem:** Currently, the codebase has no mechanism for memory-efficient metric tracking. It appends every iteration's loss to a standard Python list, causing a linear memory leak over time.
    *   **Action:** Modify `PairedTrainer.initialize_dict()` and `PairedTrainer.train()`.
    *   **Change:** Implement a custom `CircularBuffer` or `RunningAverageMeter` in `utils/metric_tracker.py`. Replace growing lists (`self.losses_dict[key].append(value)`) with these fixed-size structures. 
    *   **Reason:** Prevents RAM exhaustion during long training runs (500k+ iterations).
2.  **Decouple Rendering from Compute:**
    *   **Action:** Move the periodically executed `matplotlib` rendering logic out of the main training flow.
    *   **Change:** Create a `MetricLogger` class that aggregates tensors and computes averages asynchronously.
3.  **Visdom Native Scalar Plotting:**
    *   **Action:** In `plot_utils.py`, replace `self.vis.matplot()` with `self.vis.line()`.
    *   **Change:** Pass raw scalars to Visdom for GPU-accelerated client-side rendering.

4.  **Performance Profiling:**
    *   **Action:** Integrate `torch.profiler` or basic timing decorators around visualization calls.
    *   **Goal:** Quantify the time spent in `matplotlib` rendering vs. model computation to justify the removal of CPU-bound plotting logic.

## Phase 2: Modernization & Infrastructure Migration
*Goal: Replace Visdom with Weights & Biases (WandB) for better experiment tracking and remote access.*

1.  **WandB Integration:**
    *   **Action:** Initialize `wandb` in `train_sr_main.py` using `opts.network_version` as the run name.
    *   **Implementation:** 
        ```python
        import wandb
        wandb.init(project="SR-Project", name=opts.network_version, config=vars(opts))
        ```
2.  **Unified Reporter Interface:**
    *   **Action:** Create an abstract `BaseReporter` class in `plot_utils.py`.
    *   **Implementation:** Implement `WandBReporter` and `VisdomReporter` as subclasses. This allows toggling between "Local Debug Mode" (Visdom) and "Production/Cloud Mode" (WandB).
3.  **Config Syncing:**
    *   **Action:** Automatically log the `ConfigHolder` dictionary to WandB at start-up to ensure hyperparameter reproducibility.

## Phase 3: SR-Specific Visualization Enhancements
*Goal: Provide the "Why" behind model performance, not just the "What".*

1.  **Image Comparison Suite:**
    *   **Action:** Modify `visdom_visualize`.
    *   **Requirement:** Log a 3-image panel for both Train and Test sets:
        *   **LR (Input)** - Upsampled via Bicubic for size parity.
        *   **SR (Output)** - The model's prediction.
        *   **HR (Ground Truth)** - The target image.
    *   **WandB Tooling:** Use `wandb.Image` with captioning to enable side-by-side comparison sliders.
2.  **Edge Error Mapping:**
    *   **Action:** Add a `plot_error_map(pred, target)` method.
    *   **Logic:** Compute `torch.abs(pred - target).mean(dim=1)` and apply a heatmap (e.g., 'jet' or 'inferno').
    *   **Reason:** Helps identify if the model is failing on texture (stochastic) or edges (structural).
3.  **Quantitative Validation (Per-Batch/Epoch):**
    *   **Action:** Integrate `torchmetrics` for:
        *   **PSNR** (Peak Signal-to-Noise Ratio).
        *   **SSIM** (Structural Similarity Index).
        *   **LPIPS** (Learned Perceptual Image Patch Similarity).
    *   **Reason:** L1 loss often goes down while visual quality (LPIPS) stagnates; tracking both is critical for GAN-based SR.

## Phase 4: Execution Strategy for AI Agent
*Concrete steps to be performed:*

1.  **Step 1:** Create `utils/metric_tracker.py` to handle the `RunningAverage` logic.
2.  **Step 2:** Update `utils/plot_utils.py` to include `WandBReporter` class logic.
3.  **Step 3:** Refactor `train_sr_main.py` to initialize the reporter and log global configurations.
4.  **Step 4:** Insert PSNR/SSIM calculation logic into `paired_trainer.py`'s `test` function.
