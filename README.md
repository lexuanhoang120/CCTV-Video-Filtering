# CCTV Video Filtering Module

> **Note:** This project was completed during an internship at [PIASpace](https://www.pia.space/). Source code is not publicly available due to confidentiality restrictions.

A VLM-powered video classification module that automatically identifies CCTV and surveillance footage using frame-level analysis with InternVL and Qwen-VL models.

- **Models**: InternVL3-1B, InternVL3-2B, Qwen3-VL-2B
- **Input**: Single videos or batch JSONL datasets
- **Output**: JSONL predictions with class labels, confidence scores, and reasoning
- **Target Anomaly Classes**: Fire, Smoke, Violence, Falldown

> The first half of this README covers the **Filtering Module** in detail. Scroll past the divider for the **full pipeline** (Downloader → Filtering → Refinement).

## Demo Samples

### Module Overview

![Module Overview](docs/pipeline.png)

### Input Example

**Input video sample:** [YouTube demo](https://www.youtube.com/watch?v=yjukCBvGYSE)

**Prompt Example:**


![Prompt Example](docs/prompt.png)

Prompt design had a strong impact on accuracy — optimized prompts incorporating CCTV-specific terminology (surveillance camera, security footage, fixed camera, incident context) significantly improved classification performance.

### Output Example

![Output Example](docs/output.png)

### Lightweight Baseline

A **grayscale-difference heuristic** was also tested as a fast, non-VLM baseline: frames with high pixel-level change over time are more likely to originate from static surveillance cameras. While useful for pre-filtering, the VLM approach ultimately delivered higher accuracy.

---

## Performance

| Dataset | Model | Videos | Accuracy |
|---|---|---|---|
| Small labeled test set | Qwen3-VL-2B | 18 | **94.44%** |
| Large-scale internal dataset | InternVL3-2B | 425 | **83.53%** |

- **Qwen3-VL-2B** achieved the highest single accuracy on the curated test set with optimized prompts.
- **InternVL3-2B** performed best on the larger, real-world internal dataset.
- Prompt engineering was critical — tailored prompts incorporating surveillance-specific vocabulary consistently outperformed generic ones.

## Prerequisites

A GPU with sufficient VRAM and the following dependencies:

```bash
pip install torch torchvision opencv-python numpy pillow pyyaml transformers accelerate
```

The `base_model.py` file containing InternVLModel and QwenVLModel class definitions is also required (included in `filtering_module/`).

## Project Structure

```plaintext
.
├── filtering_module/
│   ├── base_model.py       # Model wrappers for InternVL/Qwen
│   └── main.py             # Main execution script
├── config/
│   └── config.yaml         # Configuration parameters
├── prompt/                 # The instruction sent to the model
├── dataset.jsonl           # List of videos and ground truth
├── videos/                 # Folder containing .mp4/.avi files
└── results/                # Generated JSONL prediction logs
```

## Configuration

### 1. config.yaml

```yaml
model_type: "both"          # Options: internvl, qwenvl, both
output_mode: json           # Options: json, text
num_frames: 8
prompt_path: "prompt.txt"
dataset_path: "dataset.jsonl"
video_folder: "videos"

models:
  - name: "InternVL3-1B"
    type: "internvl"
    path: "path/to/internvl/model"
  - name: "Qwen3-VL-2B"
    type: "qwenvl"
    path: "path/to/qwen/model"
```

### 2. dataset.jsonl

```json
{"video": "sample1.mp4", "is_cctv": true}
{"video": "sample2.mp4", "is_cctv": false}
```

### 3. Prompt

The prompt instructs the model to return a JSON object. For best results, include CCTV-specific terminology:

```
Analyze these frames from a video. Determine if this is CCTV/surveillance footage.
Look for indicators such as: fixed camera angle, surveillance camera placement,
security footage characteristics, timestamp overlays, and static perspective.
Return a JSON object with keys: 'class' (either 'cctv' or 'non-cctv'),
'confidence' (0-1), and 'reasoning' (string).
```

## Usage

```bash
python main.py --config config/config.yaml
```

### Command Line Arguments

| Argument | Default | Description |
|---|---|---|
| `--config` | `config/config.yaml` | Path to YAML configuration file |
| `--quiet` / `-q` | `False` | Reduce output to final summaries only |

## Output & Results

The script provides detailed terminal output for every video processed:

```plaintext
======================================================================
CCTV DETECTION RESULT - InternVL3-1B
======================================================================
Video File:        office_cam.mp4
Resolution:        1920x1080
Is CCTV:           True
Confidence:        0.95
Correctness:       ✓
Model Response:    {'class': 'cctv', 'confidence': 0.95, 'reasoning': '...'}
----------------------------------------------------------------------
PERFORMANCE METRICS
Preprocessing:         0.45s  ( 12.1%)
Model Inference:       3.20s  ( 87.9%)
======================================================================
```

Results are also saved to `results_[model_name].jsonl` containing all predictions, ground truths, and timing data for further analysis.

---

# Automatic CCTV Video Dataset Construction

## Description

This project provides an end-to-end pipeline for automatically collecting, filtering, and refining CCTV anomaly video datasets. It consists of three modules:

1. **Downloader** — generates search queries with CCTV terminology and downloads candidate videos from the web
2. **Filtering** — classifies videos as CCTV or non-CCTV using VLMs (see above)
3. **Refinement** — splits full-length CCTV videos into short anomaly clips (1–5 seconds) using VQA-based frame detection

The pipeline targets four anomaly classes — **Fire, Smoke, Violence, and Falldown** — producing 20–30 high-quality clips per class for downstream computer vision research.

## Installation

```bash
# Create environment with Python 3.10+
conda create -n piaspace python=3.10
conda activate piaspace

# Install PyTorch with CUDA support (adjust version as needed)
conda install pytorch torchvision pytorch-cuda=12.4 -c pytorch -c nvidia

# Install all project dependencies
pip install -r requirements/all.txt

# Optional: flash attention for filtering module
pip install --extra-index-url https://miropsota.github.io/torch_packages_builder flash_attn==2.8.3+pt2.6.0cu124
```

For OpenAI-powered search query generation, set your API key:

```bash
export OPENAI_API_KEY='your-api-key-here'
```

## How to Use

Before running, adjust hyperparameters in the configuration files under `assets/cfg/` (see [Configuration & Hyperparameters](#configuration--hyperparameters)).

### Complete Pipeline

```bash
bash scripts/all.sh
```

Runs downloader → filtering → refinement sequentially.

### Module 1: Downloader

```bash
bash scripts/run_downloader.sh
```

Searches and downloads videos from YouTube, Google, or other sources using LLM-generated queries enhanced with CCTV-specific terminology.

### Module 2: Filtering

```bash
bash scripts/filtering_module.sh
```

Samples frames from downloaded videos and classifies them as CCTV or non-CCTV using InternVL or Qwen-VL models. Outputs classification results with confidence scores and reasoning.

### Module 3: Refinement

```bash
bash scripts/refinement_module.sh
```

Samples 64 frames per CCTV video, uses VQA to identify frames containing the target anomaly (fire, smoke, violence, or falldown), maps frame IDs to timestamps, and cuts 1–5 second clips around detected moments.

## Configuration & Hyperparameters

All modules are configured via YAML files in `assets/cfg/`. Edit these before running. Command-line arguments override config file values.

### Downloader Module (`assets/cfg/downloader_module/config.yaml`)

| Parameter | Type | Description |
|---|---|---|
| `use_llm` | bool | Use LLM to auto-generate search queries (set `false` for manual queries) |
| `llm.backend` | str | LLM backend: `openai` or `qwen` |
| `llm.model` | str | Model path (e.g., `gpt-4o-mini`, `Qwen/Qwen2.5-1.5B-Instruct`) |
| `search_queries` | list | Manual search queries when `use_llm` is false |
| `search_queries_file` | str | Path to file with queries (one per line) |
| `search_queries_per_class` | int | Queries per class when using LLM (default: 5) |
| `data_source` | str | Video source: `youtube`, `google`, `ddgs`, or `browser_use` |
| `search.videos_per_keyword` | int | Videos to download per query (default: 10) |
| `search.max_results` | int | Max results per query |
| `download.output_dir` | str | Output directory for downloaded videos |
| `class_name` | list | Anomaly classes: `["Fire", "Violence", "Smoke", "Falldown"]` |

### Filtering Module (`assets/cfg/filtering_module/config4.yaml`)

| Parameter | Type | Description |
|---|---|---|
| `model_type` | str | VLM(s) to use: `internvl`, `qwenvl`, or `both` |
| `models` | list | Model entries with `name`, `type`, and `path` fields |
| `num_frames` | int | Frames to extract per video (default: 8) |
| `importance_sampling` | bool | Use importance-based frame sampling |
| `prompt_path` | str | Path to prompt file for CCTV classification |
| `output_mode` | str | Output format: `json` or `text` |
| `output_file` | str | Path to save classification results |
| `mode` | str | `test` (from downloader_folder) or `eval` (from dataset) |
| `downloader_folder` | str | Input folder for downloaded videos (mode=test) |
| `dataset_path` | str | Path to dataset JSONL (mode=eval) |
| `video_folder` | str | Folder containing video files (mode=eval) |

### Refinement Module (`assets/cfg/refinement_module/exp_1.yaml`)

| Parameter | Type | Description |
|---|---|---|
| `refinement_module.enabled` | bool | Enable or disable refinement |
| `refinement_module.input_dir` | str | Input path (filtering module output JSONL) |
| `refinement_module.output_dir` | str | Output directory for refined clips |
| `sampling.num_frames` | int | Frames to extract for VQA (default: 64) |
| `sampling.chunk_size` | int | Chunk size for processing (default: 16) |
| `vqa.max_new_tokens` | int | Max tokens for VQA output (default: 512) |
| `vqa.output_mode` | str | VQA output mode — `indices_scores_v4` recommended |
| `vqa.score_threshold` | float | Score threshold for detection (default: 0.3) |
| `refinement_module.N` | float | Padding seconds before/after detected segments (default: 0.0) |
