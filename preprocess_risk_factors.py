
# -*- coding: utf-8 -*-
"""
心力衰竭临床记录数据集：
数据质量检查、描述性统计、相关性热力图、
Welch T 检验、随机森林特征重要性分析。

输入：
    heart_failure_clinical_records_dataset.csv

输出：
    outputs/tables/describe.csv
    outputs/tables/missing_values.csv
    outputs/tables/iqr_outliers.csv
    outputs/tables/t_test_results.csv
    outputs/tables/rf_feature_importance.csv
    outputs/figures/correlation_heatmap.png
    outputs/figures/rf_feature_importance.png
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib

# 使用非交互式后端，便于在脚本中直接保存图片
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import ttest_ind
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


# ==========================
# 1. 路径与参数设置
# ==========================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "heart_failure_clinical_records_dataset.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "DEATH_EVENT"
RANDOM_STATE = 42

# True：对连续特征进行 IQR 缩尾，降低极端值对分析的影响。
# False：仅标记 IQR 异常值，不自动修改数值。
# 医学研究建议先检查异常记录，再决定是否缩尾。
APPLY_IQR_WINSORIZATION = False

# 绘图风格与中文字体设置
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["font.sans-serif"] = [
    "SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False


# ==========================
# 2. 读取数据与基础检查
# ==========================

def load_data():
    """读取数据，并检查关键字段。"""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"找不到数据文件：{DATA_PATH}\n"
            "请将 CSV 文件放在脚本所在目录，或修改 DATA_PATH。"
        )

    df = pd.read_csv(DATA_PATH)

    print("=" * 60)
    print("1. 数据基本信息")
    print("=" * 60)

    print(f"数据行数：{df.shape[0]}")
    print(f"数据列数：{df.shape[1]}")
    print("\n字段名称：")
    print(df.columns.tolist())

    print("\n数据类型与非空数量：")
    df.info()

    if TARGET not in df.columns:
        raise ValueError(f"数据中缺少目标字段：{TARGET}")

    # 将目标变量转为数值，非法值会变成 NaN，随后检查
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    if df[TARGET].isna().any():
        raise ValueError(
            "目标变量 DEATH_EVENT 存在缺失或非数值记录，"
            "请先核实这些记录，不能直接用于监督学习。"
        )

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("DEATH_EVENT 应仅包含 0 和 1。")

    print("\n目标变量分布：")
    print(df[TARGET].value_counts().sort_index())
    print("\n目标变量比例：")
    print(df[TARGET].value_counts(normalize=True).sort_index())

    print("\n重复行数量：")
    print(df.duplicated().sum())

    # 不在这里直接删除重复行，因为重复记录也可能需要结合患者 ID 核实
    return df


# ==========================
# 3. 描述性统计与缺失值检查
# ==========================

def descriptive_statistics(df):
    """输出描述性统计与缺失值报告。"""

    print("\n" + "=" * 60)
    print("2. 描述性统计")
    print("=" * 60)

    # describe() 默认展示数值字段的计数、均值、标准差、分位数等
    describe_df = df.describe(include="all").T

    print(describe_df)
    describe_df.to_csv(
        TABLE_DIR / "describe.csv",
        encoding="utf-8-sig"
    )

    missing_count = df.isna().sum()
    missing_ratio = df.isna().mean() * 100

    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percent": missing_ratio
    })

    print("\n缺失值检查：")

    if missing_count.sum() == 0:
        print("明确说明：数据集不存在缺失值。")
    else:
        print("发现缺失值，具体情况如下：")
        print(missing_df[missing_df["missing_count"] > 0])

    missing_df.to_csv(
        TABLE_DIR / "missing_values.csv",
        encoding="utf-8-sig"
    )

    return df


# ==========================
# 4. 医学合理性检查与 IQR 异常值检测
# ==========================

def detect_outliers(df):
    """
    检查明显不合理的医学取值，并用 IQR 标记统计异常值。

    注意：
    IQR 异常不等于数据错误。
    不自动删除所有异常值，以免抹去真实的高风险临床信号。
    """

    print("\n" + "=" * 60)
    print("3. 异常值检查")
    print("=" * 60)

    df = df.copy()

    # 以下为明显的编码或物理范围检查。
    # 发现这些记录时先报错，避免未经核实就擅自修改。
    binary_cols = [
        c for c in [
            "anaemia", "diabetes", "high_blood_pressure",
            "sex", "smoking", TARGET
        ] if c in df.columns
    ]

    for col in binary_cols:
        invalid = ~df[col].isin([0, 1]) & df[col].notna()

        if invalid.any():
            print(f"警告：{col} 有 {invalid.sum()} 条非 0/1 记录。")

    # 这些范围只用于标记明显可疑记录，不代表诊断标准。
    domain_checks = {
        "age": lambda s: s <= 0,
        "ejection_fraction": lambda s: (s <= 0) | (s > 100),
        "serum_sodium": lambda s: s <= 0,
        "serum_creatinine": lambda s: s <= 0,
        "platelets": lambda s: s < 0,
        "creatinine_phosphokinase": lambda s: s < 0,
        "time": lambda s: s < 0,
    }

    for col, rule in domain_checks.items():
        if col not in df.columns:
            continue

        values = pd.to_numeric(df[col], errors="coerce")
        bad_mask = rule(values) & values.notna()

        if bad_mask.any():
            print(
                f"医学/取值范围警告：{col} 有 "
                f"{bad_mask.sum()} 条可疑记录，行索引如下："
            )
            print(df.index[bad_mask].tolist()[:20])

    # IQR 检测仅用于数值特征，不包括二元目标变量
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in numeric_cols if c != TARGET]

    outlier_records = []

    for col in feature_cols:
        series = df[col].dropna()

        if series.empty:
            continue

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1

        # IQR 为 0 时仍按公式检查，但可能对离散变量过于敏感
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        mask = df[col].notna() & (
            (df[col] < lower) | (df[col] > upper)
        )

        outlier_records.append({
            "feature": col,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "lower_bound": lower,
            "upper_bound": upper,
            "outlier_count": int(mask.sum()),
            "outlier_percent": float(mask.mean() * 100),
        })

        if mask.any():
            print(
                f"{col}: 检测到 {mask.sum()} 条 IQR 异常值，"
                f"范围外记录占 {mask.mean() * 100:.2f}%"
            )

    outlier_df = pd.DataFrame(outlier_records)

    outlier_df.to_csv(
        TABLE_DIR / "iqr_outliers.csv",
        index=False,
        encoding="utf-8-sig"
    )

    if outlier_df.empty or outlier_df["outlier_count"].sum() == 0:
        print("IQR 检查未发现统计异常值。")
    else:
        print("\nIQR 异常值汇总：")
        print(outlier_df.sort_values(
            "outlier_count", ascending=False
        ).to_string(index=False))

    # 默认保留临床上可能合理的极端值。
    # 如明确选择缩尾，仅处理 IQR 范围外的连续变量。
    if APPLY_IQR_WINSORIZATION:
        exclude_from_winsor = set(binary_cols) | {TARGET}

        for col in feature_cols:
            if col in exclude_from_winsor:
                continue

            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            df[col] = df[col].clip(lower=lower, upper=upper)

        print(
            "\n已按 IQR 边界对连续特征进行缩尾。"
            "请在论文中说明处理规则，并与未缩尾结果进行敏感性比较。"
        )
    else:
        print(
            "\n默认不自动删除或修改 IQR 异常值。"
            "请结合原始记录和医学合理性复核后决定是否处理。"
        )

    return df


# ==========================
# 5. 相关性热力图
# ==========================

def plot_correlation_heatmap(df):
    """绘制并保存高分辨率相关性热力图。"""

    print("\n" + "=" * 60)
    print("4. 绘制相关性热力图")
    print("=" * 60)

    numeric_df = df.select_dtypes(include=np.number)

    # 使用 Spearman 相关系数，适合探索单调关联
    corr = numeric_df.corr(method="spearman")

    fig, ax = plt.subplots(figsize=(12, 10))

    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )

    ax.set_title(
        "Heart Failure Dataset: Spearman Correlation",
        fontsize=14,
        pad=16
    )

    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.tight_layout()

    save_path = FIGURE_DIR / "correlation_heatmap.png"

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"热力图已保存：{save_path}")


# ==========================
# 6. Welch 独立样本 T 检验
# ==========================

def perform_t_tests(df):
    """
    比较死亡组与存活组的连续特征。
    使用 Welch T 检验，不要求两组方差相等。

    T 检验结果反映组间均值差异，
    不代表调整混杂因素后的独立危险因素效应。
    """

    print("\n" + "=" * 60)
    print("5. 死亡组与存活组的 T 检验")
    print("=" * 60)

    # 排除目标变量和二元分类变量
    binary_cols = {
        "anaemia", "diabetes", "high_blood_pressure",
        "sex", "smoking", TARGET
    }

    continuous_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in binary_cols
    ]

    results = []

    for col in continuous_cols:
        alive = df.loc[df[TARGET] == 0, col].dropna()
        death = df.loc[df[TARGET] == 1, col].dropna()

        if len(alive) < 2 or len(death) < 2:
            print(f"{col}: 有效样本不足，跳过检验。")
            continue

        # Welch 独立样本 T 检验
        t_stat, p_value = ttest_ind(
            death,
            alive,
            equal_var=False,
            nan_policy="omit"
        )

        results.append({
            "feature": col,
            "alive_n": len(alive),
            "death_n": len(death),
            "alive_mean": alive.mean(),
            "death_mean": death.mean(),
            "mean_difference_death_minus_alive": (
                death.mean() - alive.mean()
            ),
            "t_statistic": t_stat,
            "p_value": p_value,
        })

    result_df = pd.DataFrame(results)

    if result_df.empty:
        print("没有足够的数据完成 T 检验。")
        return result_df

    # Benjamini-Hochberg 多重检验校正
    p = result_df["p_value"].to_numpy()
    m = len(p)
    order = np.argsort(p)
    ranked_p = p[order]

    adjusted_sorted = ranked_p * m / np.arange(1, m + 1)
    adjusted_sorted = np.minimum.accumulate(
        adjusted_sorted[::-1]
    )[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

    adjusted_p = np.empty(m)
    adjusted_p[order] = adjusted_sorted

    result_df["p_value_BH"] = adjusted_p

    result_df = result_df.sort_values("p_value")

    print(
        result_df[
            [
                "feature",
                "alive_mean",
                "death_mean",
                "mean_difference_death_minus_alive",
                "t_statistic",
                "p_value",
                "p_value_BH",
            ]
        ].to_string(index=False)
    )

    print(
        "\n说明：p 值反映组间均值差异的统计证据；"
        "若要估计调整后的关联效应，应进一步拟合多变量 Logistic 回归。"
    )

    result_df.to_csv(
        TABLE_DIR / "t_test_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return result_df


# ==========================
# 7. 随机森林特征重要性
# ==========================

def random_forest_importance(df):
    """
    使用训练集训练随机森林，并输出特征重要性排名。

    注意：
    time 如果是实际随访时长，可能包含预测时点之后的信息。
    本脚本默认将 time 排除在入组时风险预测的特征之外。
    """

    print("\n" + "=" * 60)
    print("6. 随机森林特征重要性分析")
    print("=" * 60)

    # 预测特征：排除目标变量
    # 若 time 为预测时点可获得的有效基线变量，
    # 且任务定义允许使用，才应考虑将其纳入。
    exclude_cols = {TARGET, "time"}

    feature_cols = [
        c for c in df.select_dtypes(include=np.number).columns
        if c not in exclude_cols
    ]

    if not feature_cols:
        raise ValueError("没有可用于训练的数值特征。")

    X = df[feature_cols].copy()
    y = df[TARGET].copy()

    # 对于本示例的少量临床记录，使用分层留出集。
    # 中位数填补器只在训练集上拟合，避免信息泄漏。
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y
    )

    imputer = SimpleImputer(strategy="median")

    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=feature_cols,
        index=X_train.index
    )

    X_test_imp = pd.DataFrame(
        imputer.transform(X_test),
        columns=feature_cols,
        index=X_test.index
    )

    # 树模型（RF / XGBoost）通常不需要标准化；
    # LR 和 MLP 通常需要 StandardScaler 等标准化处理。
    # 若后续比较 LR、MLP，应在训练集内通过 Pipeline 标准化，
    # 不要先对全体数据拟合标准化器。

    model = RandomForestClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train_imp, y_train)

    # 评估测试集 AUC，仅作为本次划分的初步参考
    if y_test.nunique() == 2:
        probabilities = model.predict_proba(X_test_imp)[:, 1]
        auc = roc_auc_score(y_test, probabilities)
        print(f"随机森林测试集 ROC-AUC（单次划分）：{auc:.3f}")

    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    importance_df["rank"] = np.arange(1, len(importance_df) + 1)

    importance_df = importance_df[
        ["rank", "feature", "importance"]
    ]

    print("\n随机森林特征重要性排名：")
    print(importance_df.to_string(index=False))

    importance_df.to_csv(
        TABLE_DIR / "rf_feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # 绘制特征重要性图
    plot_df = importance_df.head(10).sort_values(
        "importance", ascending=True
    )

    fig, ax = plt.subplots(figsize=(9, 6))

    ax.barh(
        plot_df["feature"],
        plot_df["importance"]
    )

    ax.set_xlabel("Random Forest Feature Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 10 Feature Importances")

    fig.tight_layout()

    save_path = FIGURE_DIR / "rf_feature_importance.png"

    fig.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\n特征重要性图已保存：{save_path}")

    top3 = importance_df.head(3)

    print("\n" + "=" * 60)
    print("随机森林排名前三的预测重要性特征")
    print("=" * 60)

    for _, row in top3.iterrows():
        print(
            f"{int(row['rank'])}. {row['feature']} "
            f"(importance = {row['importance']:.4f})"
        )

    print(
        "\n医学解释提醒：以上是模型预测重要性排名，"
        "不是经过混杂因素调整的独立危险因素排名，"
        "也不能单独用于推断因果关系。"
    )

    return model, importance_df


# ==========================
# 8. 主流程
# ==========================

def main():
    print("开始心力衰竭数据预处理与风险因子检测……")

    df = load_data()

    df = descriptive_statistics(df)

    df = detect_outliers(df)

    plot_correlation_heatmap(df)

    t_test_results = perform_t_tests(df)

    rf_model, importance_df = random_forest_importance(df)

    print("\n" + "=" * 60)
    print("全部分析流程执行完毕")
    print("=" * 60)

    print(f"表格目录：{TABLE_DIR}")
    print(f"图片目录：{FIGURE_DIR}")

    print(
        "\n注意：IQR 异常值默认只标记、不自动修改；"
        "如果启用了缩尾，热力图和统计检验会基于缩尾后的数据。"
    )


if __name__ == "__main__":
    main()
