import dagshub
import os, json
import numpy as np
import pandas as pd
import mlflow, mlflow.sklearn
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, roc_auc_score, classification_report
)

def load_processed_data():
    local_train = os.path.join("preprocess_dataset", "train_preprocessed.csv")
    local_test  = os.path.join("preprocess_dataset", "test_preprocessed.csv")
    root_train  = os.path.join("..", "preprocess_dataset", "train_preprocessed.csv")
    root_test   = os.path.join("..", "preprocess_dataset", "test_preprocessed.csv")

    if os.path.exists(local_train) and os.path.exists(local_test):
        train_p, test_p = local_train, local_test
    elif os.path.exists(root_train) and os.path.exists(root_test):
        train_p, test_p = root_train, root_test
    else:
        raise FileNotFoundError(
            "train_preprocessed.csv / test_preprocessed.csv tidak ditemukan di "
            "./preprocess_dataset atau ../preprocess_dataset"
        )

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
        plt.plot([0,1],[0,1],"--", color="gray")
    else:
        classes = np.unique(y_true)
        y_true_bin = label_binarize(y_true, classes=classes)
        roc_aucs = []
        for i, c in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_score[:, i])
            roc_aucs.append(auc(fpr, tpr))
            plt.plot(fpr, tpr, label=f"class {c} (AUC={roc_aucs[-1]:.3f})")
        roc_auc = float(np.mean(roc_aucs))
        plt.plot([0,1],[0,1],"--", color="gray")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curve"); plt.legend(); plt.tight_layout()
    fig.savefig(path_png, dpi=150, bbox_inches="tight"); plt.close(fig)
    return roc_auc

def main():
    use_dagshub = bool(os.getenv("MLFLOW_TRACKING_URI")) and bool(os.getenv("MLFLOW_TRACKING_USERNAME"))
    if use_dagshub:
        try:
            OWNER = os.getenv("DAGSHUB_OWNER", "189nrahfi") # Change the username
            REPO  = os.getenv("DAGSHUB_REPO",  "sistem-machine-learning") # Change the repository name
            dagshub.init(repo_owner=OWNER, repo_name=REPO, mlflow=True)
        except ImportError:
            raise RuntimeError("Package 'dagshub' belum terpasang. Tambahkan 'pip install dagshub' bila perlu.")
    else:
        os.environ.pop("MLFLOW_TRACKING_URI", None)

    X_train, X_test, y_train, y_test, n_classes = load_processed_data()
    avg = "binary" if n_classes == 2 else "macro"

    h = dict(units1=64, units2=32, activation="relu", alpha=1e-4,
             learning_rate=1e-3, batch_size=64, epochs=200)
    for pth in ["best_params.json", os.path.join("..", "best_params.json"), os.path.join("MLProject", "best_params.json")]:
        if os.path.exists(pth):
            try:
                loaded = json.load(open(pth))
                alias = {"lr": "learning_rate", "max_iter": "epochs", "hidden1": "units1", "hidden2": "units2"}
                for k, v in loaded.items():
                    h[alias.get(k, k)] = v
                print(f"[INFO] Using best_params.json: {h}  (path: {pth} )")
                break
            except Exception as e:
                print("[WARN] Failed to Read best_params.json:", pth, e)

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

    os.environ.pop("MLFLOW_RUN_ID", None)
    os.environ.pop("MLFLOW_EXPERIMENT_ID", None)

    run_ctx = mlflow.start_run(run_name="MLP-Final")

    os.makedirs("artifacts", exist_ok=True)
    with run_ctx:
        run_id = mlflow.active_run().info.run_id
        with open("artifacts/run_id.txt", "w") as f:
            f.write(run_id)
        print(f"[RUN_ID]{run_id}")

        mlflow.set_tag("phase", "final_training")
        mlflow.set_tag("rubric_level", "advanced")
        mlflow.set_tag("logical_experiment", os.getenv("MLFLOW_EXPERIMENT_NAME", "mlp_train_ci"))
        mlflow.log_params(h)

        mlp.fit(X_tr, y_tr)

        train_acc = mlp.score(X_tr, y_tr)
        val_acc   = mlp.score(X_val, y_val)
        mlflow.log_metric("train_accuracy", float(train_acc))
        mlflow.log_metric("val_accuracy",   float(val_acc))

        y_pred  = mlp.predict(X_test)
        y_proba = mlp.predict_proba(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=avg, zero_division=0)
        rec  = recall_score(y_test, y_pred, average=avg, zero_division=0)
        f1   = f1_score(y_test, y_pred, average=avg, zero_division=0)
        if n_classes == 2:
            auc_val = roc_auc_score(y_test, y_proba[:, 1])
        else:
            auc_val = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")

        mlflow.log_metric("accuracy_test",  float(acc))
        mlflow.log_metric("precision_test", float(prec))
        mlflow.log_metric("recall_test",    float(rec))
        mlflow.log_metric("f1_test",        float(f1))
        mlflow.log_metric("roc_auc_test",   float(auc_val))

        print(f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}")
        print(f"TEST  acc={acc:.4f}  prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}  AUC={auc_val:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        plot_confusion(cm, "Confusion Matrix (test)", "artifacts/confusion_matrix_test.png")
        pd.DataFrame(cm).to_csv("artifacts/confusion_matrix_test.csv", index=False)
        mlflow.log_artifact("artifacts/confusion_matrix_test.png")
        mlflow.log_artifact("artifacts/confusion_matrix_test.csv")

        auc_plot = plot_roc_and_auc(y_test, y_proba, n_classes, "artifacts/roc_curve_test.png")
        mlflow.log_artifact("artifacts/roc_curve_test.png")

        with open("artifacts/classification_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred, zero_division=0))
        mlflow.log_artifact("artifacts/classification_report.txt")

        with open("artifacts/metrics.json", "w") as f:
            json.dump({
                "train_accuracy": float(train_acc),
                "val_accuracy":   float(val_acc),
                "accuracy_test":  float(acc),
                "precision_test": float(prec),
                "recall_test":    float(rec),
                "f1_test":        float(f1),
                "roc_auc_test":   float(auc_val),
                "roc_auc_plot":   float(auc_plot)
            }, f, indent=2)
        mlflow.log_artifact("artifacts/metrics.json")

        mlflow.sklearn.log_model(mlp, artifact_path="model")

if __name__ == "__main__":
    main()