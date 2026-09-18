"""程序：模型训练。

输入：CACHE_DIR/channel_features_train.npz（逐轴箱特征、车速、标签）。
输出：
  CACHE_DIR/cv_results.csv        各候选配置的交叉验证成绩
  CACHE_DIR/oof_predictions.csv   选定配置的折外预测（供 evaluate.py 使用）
  model/rail_model.joblib         以全部运动文件重新训练的最终模型
流程：
  1. 车速低于 LOW_SPEED_RULE_MPS 的文件由规则判为 Normal，不参与训练；
  2. 对运动文件做重复分层 K 折交叉验证；每一折内部估计传感器基线后聚合完整特征，
     各特征组合按列名从中截取；车速本身不作为特征；
  3. 比较 3 种结构/分类器组合 x 3 套车厢级特征组合，共 9 个配置；
  4. 基于重复平均后的折外概率调整判定参数，使 macro F1 最大；
  5. 在与最高分相差不超过 SELECT_TOL 的配置中，选择默认判定下 macro F1 最高者，
     以全部运动文件重新训练并保存。
"""
from __future__ import annotations

import re
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
import features

CANDIDATES = (("side", "hgb"), ("side", "logreg"), ("file", "hgb"))
STRUCTURES = tuple(dict.fromkeys(st for st, _ in CANDIDATES))
FAST_SPEED_MPS = 9.0                       # 车速捷径诊断所用的高速子集下限
SELECT_TOL = 0.02                          # 候选范围：与最高分相差不超过此值
W_GRID = np.logspace(-1, 1.5, 51)          # 文件级模型：故障类概率放大倍数
T_GRID = np.round(np.linspace(0.02, 0.98, 49), 2)   # 单侧视图模型：判为故障侧的概率阈值
CV_RESULTS = C.CACHE_DIR / "cv_results.csv"
LABELS = list(range(len(C.CLASS_NAMES)))
_CAR_SPECIFIC = re.compile(r"^c\d+_")

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# ============================================================ 数据
def load_train():
    d = np.load(C.CHANNEL_FEATURES_TRAIN)
    assert tuple(d["feature_names"]) == features.CHANNEL_FEATURE_NAMES, \
        "缓存中的特征定义与 features.py 不一致，请重新运行 build_cache.py"
    return (d["filename"].astype(str), d["speed_mps"].astype(np.float64),
            d["cf"], d["label"].astype(str), d["label_id"].astype(int))


def aggregate(structure: str, speeds, cf, baseline):
    """聚合完整特征（含全部车厢级特征）；各特征组合由 keep_columns 截取。"""
    if structure == "file":
        return features.file_level(speeds, cf, baseline, car_features="full")
    x, names, _, _ = features.side_level(speeds, cf, baseline, car_features="full")
    return x, names


def side_targets(y: np.ndarray) -> np.ndarray:
    """文件标签 -> 单侧视图标签（行顺序：文件0 Side I, 文件0 Side II, 文件1 Side I, ...）。"""
    return np.column_stack([y == 1, y == 2]).reshape(-1).astype(int)


def row_targets(structure: str, y: np.ndarray) -> np.ndarray:
    return y if structure == "file" else side_targets(y)


def keep_columns(names: list[str], car_features: str) -> np.ndarray:
    """车速本身不作为特征；按车厢级特征组合截取列。"""
    keep = []
    for i, n in enumerate(names):
        if n == "speed_mps":
            continue
        if car_features == "none" and (n.startswith("k_") or _CAR_SPECIFIC.match(n)):
            continue
        if car_features == "invariant" and _CAR_SPECIFIC.match(n):
            continue
        keep.append(i)
    return np.array(keep)


# ============================================================ 模型
def make_model(name: str):
    if name == "logreg":
        return make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000),
        )
    if name == "hgb":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=200, max_leaf_nodes=15, min_samples_leaf=5,
                l2_regularization=1.0, class_weight="balanced", random_state=C.RANDOM_SEED),
        )
    raise ValueError(name)


# ============================================================ 判定与评分
def default_param(structure: str) -> float:
    return 1.0 if structure == "file" else 0.5


def decide(structure: str, out: np.ndarray, param: float) -> np.ndarray:
    """文件级：out (n,3) 概率，故障类乘以 param 后取最大；
    单侧视图：out (n,2) 为两侧故障概率，较大者不低于 param 时判为该侧，否则 Normal。"""
    if structure == "file":
        return np.argmax(out * np.array([1.0, param, param]), axis=1)
    p1, p2 = out[:, 0], out[:, 1]
    return np.where(np.maximum(p1, p2) >= param, np.where(p1 >= p2, 1, 2), 0)


def macro_f1(y, pred) -> float:
    return f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)


def tune(structure: str, out: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    grid = W_GRID if structure == "file" else T_GRID
    d0 = default_param(structure)
    best = max(grid, key=lambda p: (macro_f1(y, decide(structure, out, p)),
                                    -abs(np.log(p / d0))))
    return float(best), macro_f1(y, decide(structure, out, best))


# ============================================================ 交叉验证
def run_cv(speeds, cf, y, configs):
    n = len(y)
    width = {"file": 3, "side": 2}
    oof = {cfg: np.full((C.CV_N_REPEATS, n, width[cfg[0]]), np.nan) for cfg in configs}
    rskf = RepeatedStratifiedKFold(n_splits=C.CV_N_SPLITS, n_repeats=C.CV_N_REPEATS,
                                   random_state=C.RANDOM_SEED)
    n_splits = C.CV_N_SPLITS * C.CV_N_REPEATS
    t0 = time.perf_counter()
    for i, (tr, va) in enumerate(rskf.split(np.zeros(n), y)):
        rep = i // C.CV_N_SPLITS
        base = features.fit_baseline(cf[tr])
        mats = {}
        for st in STRUCTURES:
            xtr, names = aggregate(st, speeds[tr], cf[tr], base)
            xva, _ = aggregate(st, speeds[va], cf[va], base)
            for cs in C.CAR_FEATURE_SETS:
                keep = keep_columns(names, cs)
                mats[(st, cs)] = (xtr[:, keep], xva[:, keep])

        for cfg in configs:
            st, mname, cs = cfg
            xtr, xva = mats[(st, cs)]
            model = make_model(mname)
            model.fit(xtr, row_targets(st, y[tr]))
            proba = model.predict_proba(xva)
            oof[cfg][rep, va] = proba if st == "file" else proba[:, 1].reshape(-1, 2)
        print(f"  fold {i + 1:>2}/{n_splits}  elapsed {time.perf_counter() - t0:6.1f} s (cumulative)",
              flush=True)
    return oof


def score_config(cfg, oof_c, y, speeds_m, y_all, moving):
    st = cfg[0]
    d0 = default_param(st)
    f1_reps = [macro_f1(y, decide(st, oof_c[r], d0)) for r in range(oof_c.shape[0])]
    avg = oof_c.mean(axis=0)
    param, f1_moving = tune(st, avg, y)
    pred = decide(st, avg, param)
    fast = speeds_m >= FAST_SPEED_MPS
    pred_all = np.zeros(len(y_all), dtype=int)
    pred_all[moving] = pred
    return {
        "structure": st, "model": cfg[1], "car_features": cfg[2],
        "f1_default_mean": np.mean(f1_reps), "f1_default_std": np.std(f1_reps),
        "param": param,
        "f1_tuned_moving": f1_moving,
        "f1_tuned_fast": macro_f1(y[fast], pred[fast]),
        "f1_tuned_all": macro_f1(y_all, pred_all),
    }


def select(results: pd.DataFrame) -> pd.Series:
    """与最高分相差不超过 SELECT_TOL 的配置中，选择默认判定下 macro F1 最高者（对阈值调整依赖最小）。"""
    best = results["f1_tuned_all"].max()
    cand = results[results["f1_tuned_all"] >= best - SELECT_TOL]
    return cand.sort_values(["f1_default_mean", "f1_tuned_all"], ascending=[False, False]).iloc[0]


# ============================================================ 主流程
def main() -> None:
    t_start = time.perf_counter()
    fnames, speeds, cf, labels, y_all = load_train()
    moving = speeds >= C.LOW_SPEED_RULE_MPS
    y, speeds_m, cf_m = y_all[moving], speeds[moving], cf[moving]
    print(f"files: {len(y_all)}  moving: {int(moving.sum())}  "
          f"low-speed (rule -> Normal): {int((~moving).sum())}")
    print("moving labels:", pd.Series(labels[moving]).value_counts().to_dict())
    print(f"per-channel features K = {features.K}")

    configs = [(st, m, cs) for st, m in CANDIDATES for cs in C.CAR_FEATURE_SETS]
    print(f"\n[cross-validation] {len(configs)} configs, "
          f"{C.CV_N_REPEATS} x {C.CV_N_SPLITS}-fold")
    oof = run_cv(speeds_m, cf_m, y, configs)

    results = pd.DataFrame([score_config(cfg, oof[cfg], y, speeds_m, y_all, moving)
                            for cfg in configs])
    results = results.sort_values("f1_tuned_all", ascending=False).reset_index(drop=True)
    results.to_csv(CV_RESULTS, index=False)
    print("\n[cv results] macro F1 (tuned = decision parameter tuned on averaged OOF, "
          "slightly optimistic)")
    print(results.round(3).to_string(index=False))

    sel = select(results)
    cfg = (sel["structure"], sel["model"], sel["car_features"])
    param = float(sel["param"])
    print(f"\n[selected] structure={cfg[0]}  model={cfg[1]}  car_features={cfg[2]}  "
          f"param={param:.3f}  f1_default_mean={sel['f1_default_mean']:.3f}  "
          f"f1_tuned_all={sel['f1_tuned_all']:.3f}")

    # ---- 选定配置的折外预测
    st = cfg[0]
    avg = oof[cfg].mean(axis=0)
    pred_m = decide(st, avg, param)
    pred_def_m = decide(st, avg, default_param(st))
    oof_df = pd.DataFrame({"filename": fnames, "label": labels, "speed_mps": speeds,
                           "low_speed": ~moving})
    prob_cols = (["p_normal", "p_side1", "p_side2"] if st == "file"
                 else ["p_side1_fault", "p_side2_fault"])
    for j, col in enumerate(prob_cols):
        oof_df[col] = np.nan
        oof_df.loc[moving, col] = avg[:, j]
    pred_all = np.zeros(len(y_all), dtype=int)
    pred_def = np.zeros(len(y_all), dtype=int)
    pred_all[moving], pred_def[moving] = pred_m, pred_def_m
    oof_df["pred_default"] = [C.CLASS_NAMES[k] for k in pred_def]
    oof_df["pred"] = [C.CLASS_NAMES[k] for k in pred_all]
    oof_df.to_csv(C.OOF_PREDICTIONS, index=False)

    # ---- 最终模型：全部运动文件
    t0 = time.perf_counter()
    base = features.fit_baseline(cf_m)
    x, names = aggregate(st, speeds_m, cf_m, base)
    keep = keep_columns(names, cfg[2])
    model = make_model(cfg[1])
    model.fit(x[:, keep], row_targets(st, y))
    bundle = {
        "structure": st, "model_name": cfg[1], "car_features": cfg[2],
        "aggregation_car_features": "full",          # 推断时先聚合完整特征，再按 keep_idx 截取
        "model": model, "baseline": base.astype(np.float32),
        "all_feature_names": names, "keep_idx": keep,
        "feature_names": [names[i] for i in keep],
        "decision_param": param,
        "low_speed_rule_mps": C.LOW_SPEED_RULE_MPS,
        "class_names": C.CLASS_NAMES,
        "channel_feature_names": features.CHANNEL_FEATURE_NAMES,
        "settings": {"AGG_STATS": C.AGG_STATS, "CONTRAST_STATS": C.CONTRAST_STATS,
                     "PAIR_STATS": C.PAIR_STATS, "CAR_STATS": C.CAR_STATS,
                     "AXLE_PAIRS": C.AXLE_PAIRS, "FREQ_BANDS_HZ": C.FREQ_BANDS_HZ,
                     "WAVELENGTH_BANDS_M": C.WAVELENGTH_BANDS_M,
                     "WELCH_NPERSEG": C.WELCH_NPERSEG},
        "cv": sel.to_dict(),
    }
    C.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, C.MODEL_FILE, compress=3)

    print(f"\n[final model] trained on {int(moving.sum())} moving files, "
          f"{len(keep)} features, {time.perf_counter() - t0:.1f} s")
    print(f"  model size : {C.MODEL_FILE.stat().st_size / 1e6:.2f} MB")
    for p in (CV_RESULTS, C.OOF_PREDICTIONS, C.MODEL_FILE):
        print(f"  saved -> {p.relative_to(C.WORKSPACE_ROOT)}")
    print(f"\nelapsed: {time.perf_counter() - t_start:.1f} s")


if __name__ == "__main__":
    main()
