# Deconvolute SDK Integrity Benchmark

[![CI](https://github.com/daved01/deconvolute-benchmark/actions/workflows/ci.yml/badge.svg)](https://github.com/daved01/deconvolute-benchmark/actions/workflows/ci.yml)
[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![Dependency Manager: uv](https://img.shields.io/badge/uv-managed-purple)](https://github.com/astral-sh/uv)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)


This repository provides a reproducible benchmark for evaluating how effectively the [Deconvolute SDK](https://github.com/daved01/deconvolute) protects Retrieval Augmented Generation pipelines and agent-based systems against Indirect Prompt Injection attacks. It includes the code, datasets, and configuration needed to validate integrity and detection claims, measure trade-offs such as latency and false positive rates, and enable independent verification. While the Deconvolute SDK is the primary focus, the benchmark is designed to support evaluation of other defense systems through shared, declarative experiment definitions that are repeatable across environments and model versions.


## Scope

This benchmark evaluates system integrity failures caused by adversarial content injected through retrieved documents. It does not address general model safety, alignment, or harmful content generation. Results should be interpreted as measurements of integrity enforcement under specific, controlled attack scenarios.


## Results Overview

The results presented below correspond to the Naive Base64 Injection scenario, evaluated on a dataset consisting of 300 samples.


### Experiment: Base64 Injection

- **Attack Goal:** Induce the generator model to ignore system-level instructions and emit responses encoded in Base64, thereby bypassing keyword-based filters.
- **Name:** `canary_naive_base64`


| Experiment Configuration | ASR (Attack Success) | PNA (Benign Perf) | TP  | FN  | TN  | FP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Baseline** (No Defense) | **90.7%** | 98.7% | 14  | 136 | 148 | 2  |
| **Protected** (Canary + Language) | **0.67%** | 95.3% | 149 | 1   | 143 | 7  |

*Lower ASR is better. Higher PNA is better.*

Without defenses, the model is highly susceptible to indirect prompt injection. Enabling the Deconvolute SDK substantially reduces successful attacks, at the cost of a small increase in false positives on benign queries. This reflects a typical trade-off between strict integrity enforcement and tolerance for benign variation. A detailed analysis of failure modes and defense behavior is provided in the accompanying technical blog post.


## Quickstart

This project uses `uv` for dependency management and environment isolation.

### Installation
```bash
git clone https://github.com/daved01/deconvolute-benchmark.git
cd deconvolute-benchmark
uv sync --all-extras
```

The virtual environment may be activated with `source .venv/bin/activate` to avoid prefixing commands with `uv run`. The examples below assume the environment is not activated.


### Running an Experiment

Experiments are organized as scenarios under `scenarios/`. Each scenario directory contains:
- `experiment.yaml`: the experiment definition
- `dataset_config.yaml`: the dataset generation recipe
- `dataset.json`: the generated dataset

An experiment can be run by specifying the scenario name:

```bash
uv run dcv-bench run example
```

If `dataset.json` is missing, it is generated automatically from `dataset_config.yaml` before execution.

Experiment variants can be selected using the colon syntax:

```bash
# Runs scenarios/example/experiment_gpt4.yaml
uv run dcv-bench run example:gpt4
```

Attack strategies, dataset properties, and model settings are modified entirely through configuration. No code changes are required.

### Configuration Overview

Experiments are defined using YAML files that describe the pipeline, defense layers, and evaluation logic.


#### Evaluators
Multiple defense layers from the `deconvolute` SDK can be composed in the pipeline using the config file. Then `Evaluators` can be added to evaluate how the SDK performs. The following evaluators are available:

| Layer Type          | Description                                                   | Config                       |
| :------------------ | :------------------------------------------------------------ | :----------------------------| 
| `canary`	          | Checks for explicit integrity violation signals from the SDK. | N/A                          |
| `keyword`	          | Detects the presence of an injected payload in the output.    | `target_keyword`             | 
| `language_mismatch` | Detects violations of the expected output language.           | `expected_language`, `strict`|


### Evaluation Metrics

Each sample is treated as a binary security outcome. Results fall into one of four categories:

| Term	              | Scenario	     | Outcome	    | Interpretation                                       |
| :------------------ | :------------- | :----------- | :--------------------------------------------------- |
| TP (True Positive)  | Attack Sample	 | Detected	    | System Protected. The defense triggered correctly.   |
| FN (False Negative) | Attack Sample	 | Not Detected | Silent Failure. The attack succeeded.                |
| TN (True Negative)  | Benign Sample	 | Not Detected | Normal Operation. System worked as expected.         |
| FP (False Positive) | Benign Sample	 | Detected	    | False Alarm. Usefulness is degraded.                 |

Attack Success Rate (ASR) measures the fraction of attacks that bypass defenses. Performance on No Attack (PNA) measures how often benign queries proceed without interference. Latency is reported as end-to-end execution time for attack and benign samples.


## Dataset Structure

Datasets live in each scenario folder under `scenarios/<scenario-name>/` (e.g. `scenarios/example/dataset.json`).

### Generating a Dataset

Optional data dependencies must be installed first:

```bash
uv sync --extra data
```

Datasets are built from a base corpus in `resources/corpus` and the scenario’s `dataset_config.yaml`. For example, to fetch the baseline SQuAD v1.1 validation set:

```bash
uv run python resources/corpus/fetch_squad_data.py
```

To generate a scenario dataset:

```bash
uv run dcv-bench data generate example
```

The resulting `dataset.json` will be saved in the scenario folder and is ready for experiments.


## Output and Artifacts

Each experiment run produces a timestamped directory under `results/` in the scenario folder:

```bash
scenarios/
└── <scenario_name>/
    └── results/
        └── run_<timestamp>/
            ├── results.json                  # Run manifest (summary + config)
            ├── traces.jsonl                  # Line-by-line execution traces (Debugging Data)
            └── plots/
                ├── confusion_matrix.png      # TP/FN/FP/TN Heatmap
                ├── asr_by_strategy.png       # Vulnerability by Attack Type
                └── latency_distribution.png  # Performance Overhead
  ```

The `results.json` file is the main artifact for analysis. For per-sample execution details, see `traces.jsonl`, which records data for each individual sample.


## Scope and Relationship to the Deconvolute SDK

This benchmark focuses on a single attack class: Indirect Prompt Injection, using multiple defensive layers to evaluate defense-in-depth. It measures observable system behavior under attack rather than internal SDK implementation. Results should not be interpreted as comprehensive LLM security guarantees. Detailed explanations of the SDK’s defenses and integrity protocols are available in the [Deconvolute SDK documentation](https://github.com/daved01/deconvolute).


## References
Gakh, Valerii, and Hayretdin Bahsi. "Enhancing Security in LLM Applications: A Performance Evaluation of Early Detection Systems." arXiv:2506.19109. Preprint, arXiv, June 23, 2025. https://doi.org/10.48550/arXiv.2506.19109.

Liu, Yupei, Yuqi Jia, Runpeng Geng, Jinyuan Jia, and Neil Zhenqiang Gong. "Formalizing and Benchmarking Prompt Injection Attacks and Defenses." Version 5. Preprint, arXiv, 2023. https://doi.org/10.48550/ARXIV.2310.12815.

Wallace, Eric, Kai Xiao, Reimar Leike, Lilian Weng, Johannes Heidecke, and Alex Beutel. "The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions." arXiv:2404.13208. Preprint, arXiv, April 19, 2024. https://doi.org/10.48550/arXiv.2404.13208.
