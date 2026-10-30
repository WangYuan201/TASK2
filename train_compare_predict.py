
# -*- coding: utf-8 -*-
"""
心力衰竭临床记录数据集：模型训练、评估对比与新患者概率预测

输入：
    heart_failure_clinical_records_dataset.csv

输出：
    outputs/tables/model_comparison.csv
    outputs/tables/cross_validation_auc.csv
    outputs/figures/roc_curves.png
    outputs/models/best_model.joblib
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from sklearn.base import clone
import joblib

# XGBoost 是独立的第三方库
from xgboost import XGBClassifier


# ==================================================
# 1. 路径与随机种子
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "heart_failure_clinical_records_dataset.csv"

OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

for folder in [TABLE_DIR, FIGURE_DIR, MODEL_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

TARGET = "DEATH_EVENT"
RANDOM_STATE = 42
TEST_SIZE = 0.20

# time 通常代表随访时长。
# 如果目标是入组时的风险预测，该字段可能包含未来信息，因此默认排除。
EXCLUDE_FEATURES = {TARGET, "time"}

sns.set_theme(style="whitegrid")
plt.rcParams["axes.unicode_minus"] = False


# ==================================================
# 2. 读取数据并准备特征
# ==================================================

def load_dataset():
    """读取数据、检查目标变量并构造特征矩阵。"""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}\n"
            "请将 CSV 文件放到脚本所在目录。"
        )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"数据缺少目标列 {TARGET}")

    # 本脚本仅对数值型临床特征建模
    y = pd.to_numeric(df[TARGET], errors="coerce")

    if y.isna().any():
        raise ValueError("目标变量存在缺失或非法值，请先检查数据。")

    if not set(y.unique()).issubset({0, 1}):
        raise ValueError("DEATH_EVENT 必须只包含 0 和 1。")

    y = y.astype(int)

    # 自动选择数值型特征，排除目标变量及随访时间
    feature_cols = [
        col for col in df.select_dtypes(include=np.number).columns
        if col not in EXCLUDE_FEATURES
    ]

    if not feature_cols:
        raise ValueError("没有可用于建模的数值型特征。")

    X = df[feature_cols].copy()

    # 将正负无穷转换为缺失值，由 Pipeline 中的填补器处理
    X = X.replace([np.inf, -np.inf], np.nan)

    # 全部为空的列无法进行中位数填补，应先排除
    all_missing = X.columns[X.isna().all()].tolist()

    if all_missing:
        print("以下特征全部缺失，将排除：", all_missing)
        X = X.drop(columns=all_missing)
        feature_cols = X.columns.tolist()

    if X.shape[1] == 0:
        raise ValueError("清除无效特征后没有可用于建模的特征。")

    print("=" * 60)
    print("数据集信息")
    print("=" * 60)
    print(f"样本数：{len(X)}")
    print(f"特征数：{X.shape[1]}")
    print("使用的特征：", feature_cols)
    print("\n目标变量分布：")
    print(y.value_counts().sort_index())
    print("\n目标变量比例：")
    print(y.value_counts(normalize=True).sort_index())

    if y.value_counts().min() < 5:
        raise ValueError(
            "少数类别不足 5 例，无法可靠执行 5 折分层交叉验证。"
        )

    return X, y, feature_cols


# ==================================================
# 3. 划分训练集与测试集
# ==================================================

def split_data(X, y):
    """按 8:2 划分，使用分层抽样保证类别比例尽量接近。"""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n" + "=" * 60)
    print("训练集与测试集划分")
    print("=" * 60)
    print(f"训练集样本数：{len(X_train)}")
    print(f"测试集样本数：{len(X_test)}")
    print(f"训练集死亡事件比例：{y_train.mean():.3f}")
    print(f"测试集死亡事件比例：{y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test


# ==================================================
# 4. 定义四种模型
# ==================================================

def build_models():
    """
    为所有模型设置缺失值填补。
    LR 和 MLP 使用 StandardScaler。
    RF 和 XGBoost 不需要标准化。
    """

    models = {
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),

        "Random Forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=400,
                max_features="sqrt",
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ]),

        "XGBoost": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=200,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_lambda=2.0,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=1
            ))
        ]),

        "MLP": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(16,),
                activation="relu",
                solver="adam",
                alpha=0.01,
                learning_rate_init=0.001,
                max_iter=1500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=RANDOM_STATE
            ))
        ])
    }

    return models


# ==================================================
# 5. 交叉验证与模型训练
# ==================================================

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    只在训练集上进行交叉验证和模型选择。
    测试集仅用于最终性能评估。
    """

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    cv_records = []
    test_records = []
    fitted_models = {}
    roc_data = {}

    print("\n" + "=" * 60)
    print("模型训练、交叉验证与测试集评估")
    print("=" * 60)

    for name, pipeline in models.items():
        print(f"\n正在训练：{name}")

        # 交叉验证的每一折都会单独拟合 Pipeline 中的预处理步骤
        cv_scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring="roc_auc",
            n_jobs=1
        )

        cv_mean = float(np.mean(cv_scores))
        cv_std = float(np.std(cv_scores, ddof=1))

        cv_records.append({
            "model": name,
            "cv_auc_mean": cv_mean,
            "cv_auc_std": cv_std,
            "cv_auc_each_fold": ", ".join(
                f"{score:.4f}" for score in cv_scores
            )
        })

        # 使用完整训练集重新训练
        fitted = clone(pipeline)
        fitted.fit(X_train, y_train)

        # 找到标签为 1 的概率列，避免依赖类别顺序的假设
        fitted_classes = fitted.named_steps["model"].classes_
        positive_index = list(fitted_classes).index(1)

        probabilities = fitted.predict_proba(X_test)[:, positive_index]
        predictions = (probabilities >= 0.5).astype(int)

        # 计算测试集分类指标
        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, zero_division=0
        )
        recall = recall_score(
            y_test, predictions, zero_division=0
        )
        f1 = f1_score(y_test, predictions, zero_division=0)
        auc = roc_auc_score(y_test, probabilities)

        test_records.append({
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "test_auc": auc,
            "cv_auc_mean": cv_mean,
            "cv_auc_std": cv_std
        })

        fpr, tpr, _ = roc_curve(y_test, probabilities)

        roc_data[name] = {
            "fpr": fpr,
            "tpr": tpr,
            "auc": auc
        }

        fitted_models[name] = fitted

        print(
            f"Accuracy={accuracy:.3f}, "
            f"Precision={precision:.3f}, "
            f"Recall={recall:.3f}, "
            f"F1={f1:.3f}, "
            f"Test AUC={auc:.3f}, "
            f"CV AUC={cv_mean:.3f} ± {cv_std:.3f}"
        )

    cv_df = pd.DataFrame(cv_records).sort_values(
        "cv_auc_mean", ascending=False
    )

    result_df = pd.DataFrame(test_records).sort_values(
        "cv_auc_mean", ascending=False
    )

    cv_df.to_csv(
        TABLE_DIR / "cross_validation_auc.csv",
        index=False,
        encoding="utf-8-sig"
    )

    result_df.to_csv(
        TABLE_DIR / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n" + "=" * 60)
    print("测试集模型比较")
    print("=" * 60)
    print(
        result_df[
            [
                "model", "accuracy", "precision", "recall",
                "f1", "test_auc", "cv_auc_mean", "cv_auc_std"
            ]
        ].round(4).to_string(index=False)
    )

    # 模型选择依据训练集 5 折交叉验证的平均 AUC，
    # 而不是依据测试集 AUC 挑选，以减少测试集选择偏差。
    best_name = cv_df.iloc[0]["model"]
    best_model = fitted_models[best_name]

    print(f"\n根据训练集交叉验证 AUC 选择的模型：{best_name}")
    print(
        "提示：测试集指标仍是单次划分结果；"
        "模型选择完成后，不要再据此反复调参。"
    )

    return result_df, roc_data, best_name, best_model


# ==================================================
# 6. 绘制并保存 ROC 曲线
# ==================================================

def plot_roc_curves(roc_data):
    """将所有模型的 ROC 曲线绘制在同一张高清图片中。"""

    fig, ax = plt.subplots(figsize=(9, 7))

    for name, values in roc_data.items():
        ax.plot(
            values["fpr"],
            values["tpr"],
            linewidth=2,
            label=f"{name} (AUC={values['auc']:.3f})"
        )

    # 随机分类器参考线
    ax.plot(
        [0, 1], [0, 1],
        linestyle="--",
        color="gray",
        label="Random classifier"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)

    fig.tight_layout()

    save_path = FIGURE_DIR / "roc_curves.png"
    fig.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"\nROC 曲线已保存：{save_path}")


# ==================================================
# 7. 构造虚拟患者并预测死亡概率
# ==================================================

def predict_new_patient(best_model, feature_cols, best_name):
    """
    使用预先定义的虚拟患者特征演示 predict_proba。
    示例数值仅用于软件流程演示，不是实际患者数据。
    """

    # 常见心力衰竭临床记录数据集的示例患者。
    # 具体值必须与数据集字段定义、单位和编码保持一致。
    patient_values = {
        "age": 75,
        "anaemia": 1,
        "creatinine_phosphokinase": 582,
        "diabetes": 0,
        "ejection_fraction": 25,
        "high_blood_pressure": 1,
        "platelets": 265000,
        "serum_creatinine": 2.0,
        "serum_sodium": 135,
        "sex": 1,
        "smoking": 0
    }

    missing_features = [
        col for col in feature_cols if col not in patient_values
    ]

    if missing_features:
        raise ValueError(
            "新患者缺少以下模型所需字段："
            f"{missing_features}\n"
            "请根据实际 CSV 字段补全 patient_values。"
        )

    # 严格按训练时的特征列及顺序构造 DataFrame
    new_patient = pd.DataFrame(
        [{col: patient_values[col] for col in feature_cols}],
        columns=feature_cols
    )

    fitted_classes = best_model.named_steps["model"].classes_
    positive_index = list(fitted_classes).index(1)

    probability = best_model.predict_proba(new_patient)[0][positive_index]
    predicted_class = int(probability >= 0.5)

    print("\n" + "=" * 60)
    print("新患者预测结果（虚拟示例）")
    print("=" * 60)
    print("\n输入的患者特征：")
    print(new_patient.to_string(index=False))

    print(f"\n使用的模型：{best_name}")
    print(f"预测为 DEATH_EVENT=0 的概率：{1 - probability:.4f}")
    print(f"预测为 DEATH_EVENT=1 的概率：{probability:.4f}")
    print(f"按 0.5 阈值输出的预测类别：{predicted_class}")

    print(
        "\n注意：预测概率是模型在当前数据分布下的估计，"
        "并非经过临床验证的个人死亡风险。"
        "0.5 只是示例分类阈值，不一定适合临床决策。"
    )


# ==================================================
# 8. 保存最终模型并运行主流程
# ==================================================

def main():
    X, y, feature_cols = load_dataset()

    X_train, X_test, y_train, y_test = split_data(X, y)

    models = build_models()

    result_df, roc_data, best_name, best_model = train_and_evaluate(
        models, X_train, X_test, y_train, y_test
    )

    plot_roc_curves(roc_data)

    # 保存模型及训练时使用的特征列，方便后续复用
    model_path = MODEL_DIR / "best_model.joblib"

    joblib.dump({
        "model": best_model,
        "model_name": best_name,
        "feature_columns": feature_cols,
        "target": TARGET,
        "random_state": RANDOM_STATE
    }, model_path)

    print(f"\n最佳模型已保存：{model_path}")

    predict_new_patient(best_model, feature_cols, best_name)

    print("\n全部模型训练与预测流程完成。")
    print(f"表格目录：{TABLE_DIR}")
    print(f"图片目录：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
