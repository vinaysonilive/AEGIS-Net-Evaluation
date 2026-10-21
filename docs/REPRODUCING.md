# Reproducing the reported results

## 0. Environment
`pip install -r requirements.txt` (Python 3.13, TensorFlow 2.20.0; reported results used 2x Tesla T4, Xeon 2.0 GHz, 4 cores). On Kaggle/Colab the tested
route is the notebook in `experiments/` (cell 0.1 installs the extra packages).

## 1. Data
Download CICIoT2023 and use the **predefined** `train.csv` (5,491,971 rows) and `test.csv` (1,176,851 rows) [dataset source/version to be filled by the authors],
plus CICIoMT2024 and CIC IoT-DIAD2024 for the transfer study (only used by the notebook, section 6).
No capture/file identifier exists, so file-grouped splits are impossible (stated as a limitation in the paper).

## 2. Command-line route (main results)
| Paper result | Command |
|---|---|
| Final model (M6), one seed, deployable artifacts | `python scripts/train_final.py --train-csv train.csv --test-csv test.csv --out results` |
| Five-seed final configuration (Table 3) | `python scripts/run_ablation.py --train-csv train.csv --test-csv test.csv --configs M6` |
| Ablation M1-M14 (Table 2), five seeds | `python scripts/run_ablation.py ... --configs M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14` |
| **Paired** differences with 95 % CIs (the manuscript could only report unpaired Welch intervals) | `python scripts/summarize.py --out results --ref M6` |
| Per-class CIs, alert precision, duplicate audit, de-duplicated scores | `python scripts/evaluate.py ... --artifacts results/release` |
| Reduced protocol used for baselines | `--epochs 5 --train-frac 0.25 --seeds 42` |

`--scaler-fit train_split` (default) fits the quantile transformer on the 64 % training split, as the manuscript states. The originally saved scaler was fitted on the whole
`train.csv` before the split (`--scaler-fit train_csv` reproduces that); the effect of this difference was not separately measured in the paper.

## 3. Notebook route (baselines, transfer, XAI, latency)
`experiments/AEGIS_Revision_Experiments.ipynb` (same code as `.py`), run sections 0 -> 9 in order; each cell header lists the reviewer comments it answers. Runs are
resumable. `PROFILE = "FAST"` is the reduced protocol, `"FULL"` the paper protocol. `TARGET_FLOAT64 = False` reproduces the transfer numbers of the paper (float32 cast of target
features, as in the original pipeline); set it to `True` for the float64 re-run.

| Reviewer comment | Where |
|---|---|
| R1-1, R2-3 factorial / multi-seed ablation | `scripts/run_ablation.py`, `scripts/summarize.py` |
| R1-2 split, duplicates | `scripts/evaluate.py`, `results/release/split_indices.npz`, notebook 4.2-4.4 |
| R1-3 feature order, layer shapes, parameters | `mappings/feature_order_46.csv`, `aegis_net/model.py`, notebook 0.6, 3.1-3.2 |
| R1-4, R2-2 few-shot rule, zero-shot vs adapted vs scratch | notebook 6.2-6.5 |
| R1-5 / R2-4 minority definition, per-class CIs, alert precision | `aegis_net/metrics.py`, `scripts/evaluate.py` |
| R1-8, R2-7 latency | notebook 8.1-8.4 |
| R1-9, R2-5 label/feature mappings | `mappings/*.csv`, notebook 6.1, 6.4, 6.6 |
| R1-10, R2-6 XAI | notebook 7.1-7.5 |
| R1-11, ED code release | this repository |
| R1-12, R2-1 reproduced baselines | notebook 5.1-5.3 |

## 4. Known issues fixed in this release
1. Raw features are kept in float64 until after the quantile transformer. Casting to float32 first lowered the saved model's macro-F1 from 0.7431 to 0.6434.
2. The 8-category map is case-insensitive (the class is spelled `SqlInjection`); the old exact-case key sent it to Benign.
3. SMOTE interpolates **all 46 features**; there is no discrete-feature mask in the reported experiments (the first manuscript draft said otherwise).
4. Few-shot draws are seeded and the evaluation split is never used as validation data.
