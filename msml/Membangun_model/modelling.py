import dagshub
import os, json
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt

from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, label_binarize

def load_processed_data():
    base = "preprocess_dataset"
    train_p = os.path.join(base, "train_preprocessed.csv")
    test_p  = os.path.join(base, "test_preprocessed.csv")
    if not (os.path.exists(train_p) and os.path.exists(test_p)):
        raise FileNotFoundError(f"Tidak temukan {train_p} / {test_p}.")

    train_df = pd.read_csv(train_p)
    test_df  = pd.read_csv(test_p)
    target_col = train_df.columns[-1]

    X_train = train_df.drop(columns=[target_col]).values.astype("float32")
    y_train = train_df[target_col].values
    X_test  = test_df.drop(columns=[target_col]).values.astype("float32")
    y_test  = test_df[target_col].values
    n_classes = len(np.unique(y_train))
    return X_train, X_test, y_train, y_test, n_classes

def plot_confusion(cm, title, path):
    import matplotlib.pyplot as plt
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
        plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    else:
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        roc_aucs = []
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_aucs.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, label=f"class {c} (AUC={roc_aucs[-1]:.3f})")
        roc_auc = np.mean(roc_aucs)
        plt.plot([0,1],[0,1],"--", color="gray")

    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight"); plt.close(fig)

    return roc_auc


def main():
    np.random.seed(42)

    OWNER, REPO = "189nrahfi", "sistem-machine-learning"
    EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "sklearn_final")
    
    dagshub.init(repo_owner=OWNER, repo_name=REPO, mlflow=True)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.sklearn.autolog(log_models=False)

    X_train, X_test, y_train, y_test, n_classes = load_processed_data()
    avg = "binary" if n_classes == 2 else "macro"

    h = dict(units1=64, units2=32, activation="relu", alpha=1e-4,
             learning_rate=1e-3, batch_size=64, epochs=200)
    if os.path.exists("best_params.json"):
        try:
            h.update(json.load(open("best_params.json")))
            print("[INFO] Using best_params.json:", h)
        except Exception as e:
            print("[WARN] Failed to Read best_params.json:", e)

    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(int(h["units1"]), int(h["units2"])),
            activation=h["activation"],
            alpha=float(h["alpha"]),
            learning_rate_init=float(h["learning_rate"]),
            batch_size=int(h["batch_size"]),
            max_iter=int(h["epochs"]),
            early_stopping=True, n_iter_no_change=10,
            random_state=42
        ))
    ])

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    with mlflow.start_run(run_name="MLP-Final"):
        mlflow.set_tag("phase", "final_training")
        mlflow.set_tag("rubric_level", "advanced")
        mlflow.log_params(h)

        mlp.fit(X_tr, y_tr)

        train_acc = mlp.score(X_tr, y_tr)
        val_acc   = mlp.score(X_val, y_val)
        mlflow.log_metric("train_accuracy", float(train_acc))
        mlflow.log_metric("val_accuracy", float(val_acc))

        y_pred = mlp.predict(X_test)
        y_proba = mlp.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)

        if n_classes == 2:
            auc_val = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

        mlflow.log_metric("accuracy_test", float(acc))
        mlflow.log_metric("precision_test", float(prec))
        mlflow.log_metric("recall_test", float(rec))
        mlflow.log_metric("f1_test", float(f1))
        mlflow.log_metric("roc_auc_test", float(auc_val))

        print(f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        print(f"test_acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  AUC={auc_val:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion(cm, "Confusion Matrix (test)", "confusion_matrix_test.png")
        mlflow.log_artifact("confusion_matrix_test.png")

        auc_from_plot = plot_roc_and_auc(y_test, y_proba, n_classes, "roc_curve_test.png")
        mlflow.log_artifact("roc_curve_test.png")

        os.makedirs("model_artifact", exist_ok=True)
        dump(mlp, "model_artifact/MLP.joblib")
        mlflow.log_artifacts("model_artifact", artifact_path="model")


if __name__ == "__main__":
    main()
