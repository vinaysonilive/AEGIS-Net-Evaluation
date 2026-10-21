"""Single source of truth for hyper-parameters (mirrors Table 1 of the manuscript and configs/default.json)."""
SEEDS = [7, 42, 100, 512, 2026]      # five seeds used for every multi-seed result
SPLIT_SEED = 42                      # shuffle seed and train/validation split seed
BATCH_SIZE = 256
EPOCHS = 15                          # maximum epochs (early stopping on validation loss)
PATIENCE = 5
LR = 1e-3                            # Adam
GAMMA, ALPHA = 2.0, 0.25             # focal loss
T_FLOOR = 3000                       # selective-SMOTE floor (classes with N_c < T_FLOOR are raised to T_FLOOR)
MINORITY_THRESHOLD = 5000            # majority/minority split by TEST support (see Section 3.7.2)
N_QUANTILES = 1000
QUANTILE_SUBSAMPLE = 100_000
LABEL_COLUMN = "label"
BENIGN = "BenignTraffic"
VAL_FRACTION = 0.2                   # 80/20 stratified split of train.csv
