# AEGIS-Net

Code for **"AEGIS-Net: An Imbalance-Aware CNN-Attention-BiLSTM Framework for Multi-Class IoT Intrusion Detection with Cross-Dataset Validation"**
(IJIES). Quantile Transformer + selective SMOTE (T_floor = 3,000) + focal loss on a Conv1D-MHA-BiLSTM classifier (210,978 parameters) for the 34-class CICIoT2023 benchmark,
with ablation, reproduced baselines, cross-dataset transfer (CICIoMT2024, CIC IoT-DIAD2024), XAI audit and latency measurements.

> Status: results in the paper were produced with the code in `experiments/` and the scripts below; see `docs/REPRODUCING.md` for the exact commands and `results_reference/` for expected numbers.

## Layout
```
aegis_net/        model, losses, data split / scaling / selective SMOTE, metrics and CIs (importable package)
scripts/          train_final.py, run_ablation.py (M1-M14, per seed), evaluate.py, summarize.py (paired CIs)
experiments/      full notebook (baselines, transfer, XAI, latency) - the code behind the baseline, transfer, XAI and latency results
mappings/         46-feature order, label and feature mappings to CICIoMT2024 and CIC IoT-DIAD2024
configs/          default.json (hyper-parameters, seeds, reduced-protocol settings)
results_reference/ reported ablation table and expected headline numbers
tests/            smoke tests on synthetic data (no dataset needed)
```

## Quick start
```bash
pip install -r requirements.txt
python scripts/train_final.py --train-csv /data/ciciot2023/train.csv --test-csv /data/ciciot2023/test.csv --out results   # M6, seed 42
python scripts/run_ablation.py  --train-csv ... --test-csv ... --configs M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12 M13 M14   # 5 seeds each
python scripts/summarize.py --out results --ref M6                                                                     # mean +/- CI, paired differences
python scripts/evaluate.py --train-csv ... --test-csv ... --artifacts results/release                                  # CIs, alert precision, duplicate audit
pytest -q                                                                                                              # smoke tests
```

## Reproducibility details
- **Seeds:** 7, 42, 100, 512, 2026 (`aegis_net/config.py`); `tf.keras.utils.set_random_seed`, TensorFlow op-determinism, SMOTE `random_state = seed`.
- **Split:** `train.csv` shuffled (seed 42) and split 80/20 stratified (seed 42) -> 4,393,576 train / 1,098,395 validation; `test.csv` (1,176,851) used once. Indices are written to `results/release/split_indices.npz`.
- **Hyper-parameters:** Adam 1e-3, batch 256, up to 15 epochs (patience 5), focal gamma 2.0 / alpha 0.25, class weights only in ablation rows M3, M4, M7, M8.
- **Schema:** exact 46-feature order in `mappings/feature_order_46.csv`; SMOTE interpolates all 46 features (no discrete mask).
- **Hardware used for the paper:** 2x Tesla T4, Xeon 2.0 GHz (4 cores); GPU results can differ slightly across hardware and drivers.

## Data
CICIoT2023, CICIoMT2024 and CIC IoT-DIAD2024 are public datasets of the Canadian Institute for Cybersecurity. The repository does not redistribute them.
https://www.unb.ca/cic/datasets/iotdataset-2023.html
https://www.unb.ca/cic/datasets/iomt-dataset-2024.html
https://www.unb.ca/cic/datasets/iot-diad-2024.html

## Citation
If you use this code or methodology, please cite:
> V. Soni, A. Jha. "AEGIS-Net: An Imbalance-Aware CNN–Attention-BiLSTM Framework for Multi-Class IoT Intrusion Detection with Cross-Dataset Evaluation," *International Journal of Intelligent Engineering & Systems*, 2026 (Under Review).
