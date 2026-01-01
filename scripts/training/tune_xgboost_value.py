import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import pandas as pd
from tqdm import tqdm

data = pd.read_csv("/home/lz80/asi_goalkeeper_positioning/stores/value_features.csv")
X = data.drop(columns=['scores_xg', 'match_id', 'frame'])
y = data['scores_xg']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def tune():
    param_space = {
        "n_estimators":  (500, 10000),
        "learning_rate": (0.01, 0.3),
        "max_depth":     (3, 10),
        "min_child_weight": (1, 10),
        "subsample":     (0.5, 1.0),
        "colsample_bytree": (0.5, 1.0),
        "reg_lambda":    (0.0, 5.0),
        "reg_alpha":     (0.0, 5.0),
        "gamma":         (0.0, 5.0),     
    }

    def sample_params(rng):
        """Sample one random hyperparameter set."""
        return {
            "learning_rate":   rng.uniform(*param_space["learning_rate"]),
            "max_depth":       rng.integers(*param_space["max_depth"]),
            "min_child_weight": rng.integers(*param_space["min_child_weight"]),
            "subsample":       rng.uniform(*param_space["subsample"]),
            "colsample_bytree": rng.uniform(*param_space["colsample_bytree"]),
            "reg_lambda":      rng.uniform(*param_space["reg_lambda"]),
            "reg_alpha":       rng.uniform(*param_space["reg_alpha"]),
            "gamma":           rng.uniform(*param_space["gamma"]),
        }


    rng = np.random.default_rng(2025)
    n_iter = 30 

    best_rmse = float("inf")
    best_params = None
    best_model = None

    for i in tqdm(range(1, n_iter + 1)):
        params = sample_params(rng)

        model = xgb.XGBRegressor(
            n_estimators=10000,
            learning_rate=params["learning_rate"],
            max_depth=params["max_depth"],
            min_child_weight=params["min_child_weight"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_lambda=params["reg_lambda"],
            reg_alpha=params["reg_alpha"],
            gamma=params["gamma"],
            early_stopping_rounds=50,
            objective="reg:squarederror",
            eval_metric="rmse",
            n_jobs=-1,
            tree_method="hist", 
        )

        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        y_pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, y_pred, squared=False)

        print(f"Training, with RMSE {rmse}, params {params}")
        if rmse < best_rmse:
            best_rmse = rmse
            best_params = params
            best_model = model

    print("\nBest RMSE:", best_rmse)
    print("Best params:", best_params)

    best_model.save_model('/home/lz80/asi_goalkeeper_positioning/stores/model/value_model.model')

def train():
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(
        n_estimators=100000,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="reg:squarederror",
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        eval_metric="rmse",
        verbose=50,
        early_stopping_rounds=50,
    )

    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    print(f"Validation RMSE: {rmse:.6f}")
    model.save_model('/home/lz80/asi_goalkeeper_positioning/stores/model/value_model_2.model')
train()