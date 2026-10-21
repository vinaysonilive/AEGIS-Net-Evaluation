# Numbers to compare against after a re-run (from the revised manuscript)

| Quantity | Value |
|---|---|
| Split | train.csv 5,491,971 -> 4,393,576 train + 1,098,395 validation; test.csv 1,176,851 |
| Saved final model (seed 42, float64 pipeline) | acc 99.10 %, macro-F1 0.7431 (95 % CI 0.7369-0.7489), MCC 0.9901 |
| Group means (18 majority / 16 minority classes by test support) | 0.9830 / 0.4732 |
| Exact duplicates | 46,231 test rows (3.93 %) also in training; de-duplicated test macro-F1 0.7436 |
| Alert precision (TPR 0.99533, FPR 0.01725) | 86.5 % at 10 %, 36.8 % at 1 %, 5.5 % at 0.1 % attack prevalence |
| Parameters | full 210,978 - no attention 144,674 - no BiLSTM 112,162 - matched 210,194 / 211,067 / MLP 211,399 |
| Reduced-protocol baselines (macro-F1) | RF 0.7745, XGBoost 0.7657, Tab-Transformer 0.7324, [9] 0.7308, [8] 0.7254, LightGBM 0.6689 |
| Zero-shot transfer | CICIoMT2024 63.84 % acc / macro-F1 0.3619; CIC IoT-DIAD2024 15.61 % / 0.0831 |
| GPU latency (T4, batch 1, model only) | median 2.81 ms; end to end 18.6 ms |

GPU results can differ slightly across hardware/driver versions even with op-determinism enabled.
