# ATML PA3: fine-tuning, visual grounding, and test-time compute

Code for the three PA3 experiment suites. Python modules hold all experiment logic. Kaggle notebooks are generated from a fixed source snapshot, and they only install the environment and launch those modules. Every GPU job runs on Kaggle T4×2. Two independent experiments share one session as GPU *lanes* (`scripts/run_lanes.py`), and jobs are spread across the configured Kaggle accounts.

| Task | Package | What it runs |
|---|---|---|
| 1. LLM fine-tuning | `task1_finetune/` | Qwen3-0.6B on 40k MetaMathQA: Full FT, Full FT + L2-SP, LoRA-4, LoRA-16, OPLoRA-16; lm-eval (GSM8K CoT, HellaSwag, BoolQ, ARC-Easy); qualitative prompts; ρk / η / R(z) diagnostics |
| 2. VLM grounding | `task2_vlm/` | Frozen SigLIP → MLP connector → Qwen3-0.6B (LoRA from stage 2); three-stage curriculum; correct / complementary / mismatched / no-image evaluation; optional 49-token run |
| 3. Test-time compute | `task3_ttc/` | Offline Best-of-N, self-consistency, self-certainty, RWC, VisConf and tokenwise fusion, adaptive stopping on the supplied rollouts; live single / parallel / refinement comparison |

## Setup

Use Python 3.11 or 3.12, run from the repository root:

```bash
python -m pip install -r requirements.txt
python -m scripts.download_assets
python -m unittest discover -s tests -v
```

The unit tests run on CPU with tiny random models. Every Kaggle job also runs them before doing any work.

- **Pinned inputs** (configured in `configs/base.yaml`): Qwen3-0.6B `c1899de2…`, MetaMathQA `aa4f34d3…`, GSM8K `740312ad…`, SigLIP `7fd15f06…`.
- **Qualitative prompts:** `first_plot_questions.yaml` comes from the public emergent-misalignment repository at commit `e40daef5…`, and its sha256 is verified. It has 8 questions in plain, `_json` and `_template` forms. The questions to annotate are the first and last in plain order: `what_is_your_wish` and `enough_of_my_husband`.
- **Course data:** put the supplied ATML-PA3-Data in `course_data/`. `python -m task3_ttc.load --inspect` prints its schema. If a column name differs from the candidates in `configs/task3.yaml` under `schema`, add it there.

## Reproducing each task

Every command resumes when re-run, and skips work that has already finished.

**Task 1.** Each lane trains, evaluates and runs diagnostics for one method.
```bash
python -m task1_finetune.data
python -m task1_finetune.evaluate --config configs/task1_full_ft.yaml --stage baseline
python -m task1_finetune.train --config configs/task1_full_ft.yaml --resume      # also l2sp, lora4, lora16, oplora16
python -m task1_finetune.evaluate --config configs/task1_full_ft.yaml --stage midpoint
python -m task1_finetune.evaluate --config configs/task1_full_ft.yaml --stage final
python -m task1_finetune.diagnostics --config configs/task1_full_ft.yaml --stage final
```

**Task 2.**
```bash
python -m task2_vlm.prepare_data       # COCO/VQAv2 selection (IDs in manifests/task2/selection.json) and image download
python -m task2_vlm.features           # SigLIP features cached in FP16; the encoder is then released
python -m task2_vlm.train --config configs/task2_vlm.yaml --resume
python -m task2_vlm.evaluate --config configs/task2_vlm.yaml
```
`configs/task2_vlm49.yaml` is the optional 49-token run.

**Task 3.**
```bash
python -m task3_ttc.load        # join, alignment report, 200/800 split and 50 live IDs
python -m task3_ttc.offline     # selection curves, bootstrap, signals, adaptive stopping
python -m task3_ttc.live        # needs live.revision set from the supplied README.MD
```

## Kaggle execution (two accounts)

- **Accounts:** `configs/kaggle_accounts.yaml` lists each account and its credential profile directory. `null` means `~/.kaggle`. The file holds no credentials and is left out of the Kaggle source snapshot.
- **Credential isolation:** for non-default accounts, the CLI wrapper `scripts/kaggle_profile_cli.py` redirects OAuth token refreshes into that account's profile.
- **Job manifest:** `experiments/runs.json` defines each job:
  - setup commands, then one lane per GPU;
  - required artifacts;
  - an optional account pin;
  - an estimated GPU-hour cost.

  Jobs gated by `depends_on: ["smoke"]` wait for the feasibility job. Jobs marked `hold` wait for inputs the user must supply.

```bash
python -m scripts.kaggle_runner list
python -m scripts.kaggle_runner quota                 # GPU-hour ledger per account / ISO week
python -m scripts.kaggle_autopilot --once --only smoke
python -m scripts.kaggle_autopilot                    # submit, poll, download, import (hourly; --interval to change)
python -m scripts.upload_course_data                  # private dataset per account, then remove `hold`
python -m scripts.validate_results                    # checks every manifest artifact and receipt exists
python -m scripts.collect_results                     # tables, figures and annotation sheet in report/generated/
```

The autopilot submits private notebooks only. Before submitting it checks two things:
- **Concurrency:** at most 2 GPU jobs and 1 CPU job per account.
- **Weekly quota:** the job's estimated hours must fit under the account's `weekly_gpu_hours`.

When a job finishes, the autopilot imports `outputs/`, `results/` and `manifests/` and writes a receipt to `results/jobs/<run>.json`. A failed session keeps its recovery files, and there are two ways to resume it:
- **After a timeout:** rebuild with `kaggle_runner build <run> --username U --resume-kernel U/<slug> --attempt 2`.
- **After an error:** Kaggle will not mount an errored notebook as a source. Instead, `kaggle_runner stage-resume <run> U/<slug> --attempt 2` copies its outputs into a private dataset. Then add an entry to `kaggle_jobs/autopilot/retries.json` with `attempt`, `resume_dataset`, and `resume_source` (the old snapshot's sha256, needed when only non-training code changed). The autopilot resubmits it on the owning account.

Large intermediates live under `remote/` and are deleted after a job succeeds: FP16 full-model snapshots, cached features, images, and recovery checkpoints. Checkpoint sizes are recorded in each `run.json`.

## Outputs

- `results/task1/`:
  - `<run>_run.json`: training time, peak memory, trainable parameters, checkpoint sizes and method overheads;
  - `<run>_train.jsonl`: per-update loss and learning rate (plus Ω for L2-SP);
  - `eval/<condition>/`: aggregate scores and per-example samples, including extracted GSM8K answers and truncation flags;
  - `qualitative/`: 24 responses per condition;
  - `diagnostics/`.
- `results/task2/<run>/`: `dev.jsonl` (learning curves), `stage_stats.json`, `connector_calibration.json`, and `final/` predictions per suite and condition, with `final_summary.json` (bootstrap intervals). Feature-caching cost is in `results/task2/feature_cache.json`.
- `results/task3/<dataset>/`: split IDs, rollout labels, per-question selections, summaries, the bootstrap file, signals, and the adaptive grid with its held-out results. Live-generation results are in `results/task3/live/`.

## Attribution

- **Kaggle job pipeline:** `scripts/kaggle_*` and `common/runtime.py` are adapted from my PA2 repository.
- **VQA answer normalization:** `task2_vlm/metrics.py` follows the rules of the official VQA evaluation script (contractions, number words, articles and punctuation).
- **MathVista normalization:** `task3_ttc/answers.py` follows the released MathVista `normalize_extracted_answer` logic.
- **Libraries:** benchmarks use EleutherAI lm-evaluation-harness 0.4.9.1, and MATH equivalence uses HuggingFace math-verify 0.9.0. LoRA initialization matches PEFT's defaults, and a unit test checks it against PEFT.
- **Methods:** OPLoRA follows Xiong and Xie, and VisConf and RWC follow Usman et al.
