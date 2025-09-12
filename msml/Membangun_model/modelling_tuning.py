import optuna
import dagshub
import os, json
import numpy as np
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt

from modelling import load_processed_data

from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize

def plot_confusion(cm, title, path):
    fig = plt.figure()

    plt.imshow(cm, interpolation="nearest")
    plt.title(title); plt.colorbar()

    ticks = np.arange(cm.shape[0]); plt.xticks(ticks, ticks); plt.yticks(ticks, ticks)

    plt.xlabel("Predicted"); plt.ylabel("True"); plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)

def plot_roc_and_auc(y_true, y_score, n_classes, path_png):
    fig = plt.figure()
    if n_classes == 2:
        fpr, tpr, _ = roc_curve(y_true, y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    else:
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        vals = []
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            vals.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, label=f"class {c} AUC={vals[-1]:.3f}")
        roc_auc = float(np.mean(vals))
        plt.plot([0,1],[0,1],"--", color="gray")

    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (val)"); plt.legend(); plt.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    return roc_auc

def main():
    np.random.seed(42)

    OWNER, REPO = "189nrahfi", "sistem-machine-learning"
    EXPERIMENT_NAME = os.getenv("MLFLOW_TUNING_EXPERIMENT", "mlp_sklearn_optuna")

    dagshub.init(repo_owner=OWNER, repo_name=REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=False)

    X_train, X_test, y_train, y_test, n_classes = load_processed_data()
    avg = "binary" if n_classes == 2 else "macro"
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "units1": trial.suggest_categorical("units1", [32, 64, 128, 256]),
            "units2": trial.suggest_categorical("units2", [16, 32, 64, 128]),
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 1e-6, 1e-2, log=True),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [32, 64, 128]),
            "epochs": trial.suggest_int("epochs", 100, 500, step=50),
        }

        with mlflow.start_run(nested=True, run_name="MLP-trial"):
            mlflow.log_params(params)

            est = Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(
                    hidden_layer_sizes=(int(params["units1"]), int(params["units2"])),
                    activation=params["activation"],
                    alpha=float(params["alpha"]),
                    learning_rate_init=float(params["learning_rate"]),
                    batch_size=int(params["batch_size"]),
                    max_iter=int(params["epochs"]),
                    early_stopping=True, n_iter_no_change=10,
                    random_state=42
                ))
            ])

            est.fit(X_tr, y_tr)
            y_pred  = est.predict(X_val)
            y_proba = est.predict_proba(X_val)

            f1  = f1_score(y_val, y_pred, average=avg, zero_division=0)
            pre = precision_score(y_val, y_pred, average=avg, zero_division=0)
            rec = recall_score(y_val, y_pred, average=avg, zero_division=0)

            val_acc = (y_pred == y_val).mean()

            if n_classes == 2:
                auc_val = roc_auc_score(y_val, y_proba[:, 1])
            else:
                auc_val = roc_auc_score(y_val, y_proba, multi_class="ovr", average="macro")

            mlflow.log_metric("f1_valid", float(f1))
            mlflow.log_metric("precision_valid", float(pre))
            mlflow.log_metric("recall_valid", float(rec))
            mlflow.log_metric("val_accuracy", float(val_acc))
            mlflow.log_metric("roc_auc_valid", float(auc_val))

            cm = confusion_matrix(y_val, y_pred)
            plot_confusion(cm, "Confusion Matrix (val)", "confusion_matrix_val.png")
            mlflow.log_artifact("confusion_matrix_val.png")

            auc_from_plot = plot_roc_and_auc(y_val, y_proba, n_classes, "roc_curve_val.png")
            mlflow.log_artifact("roc_curve_val.png")

        return f1

    storage = "sqlite:///optuna_study_mlp.db"
    study = optuna.create_study(
        study_name="mlp_tuning_f1",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
    )
    n_trials = int(os.getenv("OPTUNA_N_TRIALS", "30"))
    study.optimize(objective, n_trials=n_trials, n_jobs=1, gc_after_trial=True)

    best = study.best_params.copy()

    best.setdefault("units1", 64); best.setdefault("units2", 32)
    best.setdefault("activation", "relu"); best.setdefault("alpha", 1e-4)
    best.setdefault("learning_rate", 1e-3); best.setdefault("batch_size", 64)
    best.setdefault("epochs", 200)

    with open("best_params.json", "w") as f:
        json.dump(best, f, indent=2)
    print("[INFO] Saved Best Params:", best)
    print("Best F1:", study.best_value)

if __name__ == "__main__":
    main()