"""特征提取（纯函数，不涉及文件读写）。

输入：单个文件的转速脉冲序列 pulse (T,) 与信号张量 signals (8, 8, 2, T)。
输出：带名称的一维特征向量（文件级 1 组，单侧视图 2 组）。
由 build_cache.py（训练阶段）与 predict.py（推断阶段）共同调用，
以保证两个阶段的特征计算完全一致。
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import signal as sps
from scipy import stats

import config as C

EPS = 1e-12
AGG_STATS = ("mean", "median", "max", "std")      # 同侧 32 个轴箱上的聚合统计量
CONTRAST_STATS = ("mean", "median")               # 两侧之差所用的统计量
SIDE_TAGS = {"Side I": "s1", "Side II": "s2"}
CH_TAGS = {"vibration": "vib", "shock": "shk"}
_SIDE_POS_IDX = {"Side I": list(C.SIDE_I_POS_IDX), "Side II": list(C.SIDE_II_POS_IDX)}
_STAT_FUNCS = {"mean": np.nanmean, "median": np.nanmedian, "max": np.nanmax, "std": np.nanstd}


def _channel_feature_names() -> tuple[str, ...]:
    names = ["log_rms", "log_p2p", "crest", "kurtosis", "skewness"]
    for lo, hi in C.FREQ_BANDS_HZ:
        names += [f"logE_f{lo}_{hi}", f"relE_f{lo}_{hi}"]
    for lo, hi in C.WAVELENGTH_BANDS_M:
        tag = f"wl{round(lo * 100)}_{round(hi * 100)}cm"
        names += [f"logE_{tag}", f"relE_{tag}"]
    names += ["dom_freq_hz", "dom_wavelength_m"]
    return tuple(names)


CHANNEL_FEATURE_NAMES = _channel_feature_names()


# ============================================================ 车速
def compute_speed_mps(pulse: np.ndarray) -> float:
    """由测速齿轮脉冲的 0/1 跳变次数计算车速（m/s）。"""
    n_transitions = int(np.count_nonzero(np.diff(np.asarray(pulse))))
    duration_s = len(pulse) / C.FS_HZ
    revs_per_s = n_transitions / C.TRANSITIONS_PER_REV / duration_s
    return float(revs_per_s * C.WHEEL_CIRCUMFERENCE_M)


# ============================================================ 单通道特征
def channel_features(signals: np.ndarray, speed_mps: float) -> np.ndarray:
    """对每一路信号计算特征，返回 (8, 8, 2, K)，K = len(CHANNEL_FEATURE_NAMES)。"""
    x = np.asarray(signals, dtype=np.float64)
    x = x - x.mean(axis=-1, keepdims=True)                 # 去除直流偏置
    t = x.shape[-1]

    # ---- 时域
    rms = np.sqrt(np.mean(x * x, axis=-1))
    valid = rms > EPS
    p2p = x.max(axis=-1) - x.min(axis=-1)
    with np.errstate(all="ignore"):
        crest = np.where(valid, np.abs(x).max(axis=-1) / (rms + EPS), 0.0)
        kurt = np.where(valid, stats.kurtosis(x, axis=-1, fisher=True, bias=False), 0.0)
        skew = np.where(valid, stats.skew(x, axis=-1, bias=False), 0.0)
    feats = [np.log10(rms + EPS), np.log10(p2p + EPS), crest, kurt, skew]

    # ---- 频域（Welch 功率谱密度）
    nperseg = min(C.WELCH_NPERSEG, t)
    noverlap = min(C.WELCH_NOVERLAP, nperseg // 2)
    f, pxx = sps.welch(x, fs=C.FS_HZ, nperseg=nperseg, noverlap=noverlap, axis=-1)
    df = f[1] - f[0]
    f_min = C.FREQ_BANDS_HZ[0][0]
    total = pxx[..., f >= f_min].sum(axis=-1) * df + EPS

    def band_energy(lo: float, hi: float) -> np.ndarray:
        mask = (f >= lo) & (f < hi)
        if not mask.any():
            return np.full(rms.shape, np.nan)
        return pxx[..., mask].sum(axis=-1) * df

    for lo, hi in C.FREQ_BANDS_HZ:
        e = band_energy(lo, hi)
        feats += [np.log10(e + EPS), e / total]

    # ---- 波长域：lambda = v / f
    speed_ok = speed_mps >= C.MIN_SPEED_MPS
    for lam_lo, lam_hi in C.WAVELENGTH_BANDS_M:
        if speed_ok:
            e = band_energy(speed_mps / lam_hi, speed_mps / lam_lo)
        else:
            e = np.full(rms.shape, np.nan)
        feats += [np.log10(e + EPS), e / total]

    # ---- 主频与主波长
    search = f >= f_min
    dom_f = f[search][np.argmax(pxx[..., search], axis=-1)]
    dom_wl = speed_mps / dom_f if speed_ok else np.full(rms.shape, np.nan)
    feats += [dom_f, dom_wl]

    return np.stack(feats, axis=-1)


# ============================================================ 侧别聚合
def _side_stats(chan_feats: np.ndarray, side: str) -> dict[str, np.ndarray]:
    """在同侧 32 个轴箱上聚合，返回 {统计量: (2 通道, K)}。"""
    k = chan_feats.shape[-1]
    sub = chan_feats[:, _SIDE_POS_IDX[side]].reshape(-1, C.N_CHANNEL_TYPES, k)   # (32, 2, K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)             # 全 NaN 切片
        return {s: fn(sub, axis=0) for s, fn in _STAT_FUNCS.items()}


def _file_vector(ss: dict, speed_mps: float) -> tuple[np.ndarray, list[str]]:
    """文件级特征：车速 + 两侧聚合特征 + 两侧之差。"""
    names, vals = ["speed_mps"], [speed_mps]
    for side in C.SIDES:
        for ci, ch in enumerate(C.CHANNEL_TYPES):
            for k, fn in enumerate(CHANNEL_FEATURE_NAMES):
                for s in AGG_STATS:
                    names.append(f"{SIDE_TAGS[side]}_{CH_TAGS[ch]}_{fn}_{s}")
                    vals.append(ss[side][s][ci, k])
    for ci, ch in enumerate(C.CHANNEL_TYPES):
        for k, fn in enumerate(CHANNEL_FEATURE_NAMES):
            for s in CONTRAST_STATS:
                names.append(f"d_{CH_TAGS[ch]}_{fn}_{s}")
                vals.append(ss["Side I"][s][ci, k] - ss["Side II"][s][ci, k])
    return np.asarray(vals, dtype=np.float64), names


def _side_vector(ss: dict, speed_mps: float, side: str) -> tuple[np.ndarray, list[str]]:
    """单侧视图特征：车速 + 侧别指示 + 本侧聚合特征 + (本侧 - 对侧)。"""
    other = "Side II" if side == "Side I" else "Side I"
    names, vals = ["speed_mps", "is_side_II"], [speed_mps, float(side == "Side II")]
    for ci, ch in enumerate(C.CHANNEL_TYPES):
        for k, fn in enumerate(CHANNEL_FEATURE_NAMES):
            for s in AGG_STATS:
                names.append(f"own_{CH_TAGS[ch]}_{fn}_{s}")
                vals.append(ss[side][s][ci, k])
    for ci, ch in enumerate(C.CHANNEL_TYPES):
        for k, fn in enumerate(CHANNEL_FEATURE_NAMES):
            for s in CONTRAST_STATS:
                names.append(f"d_{CH_TAGS[ch]}_{fn}_{s}")
                vals.append(ss[side][s][ci, k] - ss[other][s][ci, k])
    return np.asarray(vals, dtype=np.float64), names


# ============================================================ 对外接口
def extract_all(pulse: np.ndarray, signals: np.ndarray) -> dict[str, tuple[np.ndarray, list[str]]]:
    """一次计算，返回 {"file": ..., "Side I": ..., "Side II": ...}，每项为 (特征值, 特征名)。"""
    v = compute_speed_mps(pulse)
    cf = channel_features(signals, v)
    ss = {side: _side_stats(cf, side) for side in C.SIDES}
    out = {"file": _file_vector(ss, v)}
    for side in C.SIDES:
        out[side] = _side_vector(ss, v, side)
    return out


def extract_file_features(pulse: np.ndarray, signals: np.ndarray) -> tuple[np.ndarray, list[str]]:
    return extract_all(pulse, signals)["file"]


def _self_check() -> None:
    import time
    import data_io

    t0 = time.perf_counter()
    pulse, sig = data_io.load_file(C.TRAIN_DIR / "Train1.csv")
    t1 = time.perf_counter()
    out = extract_all(pulse, sig)
    t2 = time.perf_counter()

    fv, fn = out["file"]
    s1v, s1n = out["Side I"]
    s2v, s2n = out["Side II"]
    assert len(fv) == len(fn) == len(set(fn))
    assert len(s1v) == len(s1n) == len(set(s1n)) and s1n == s2n
    assert not np.isinf(fv).any() and not np.isinf(s1v).any() and not np.isinf(s2v).any()

    fmap = dict(zip(fn, fv))
    for n, v in zip(s1n, s1v):
        if n.startswith("own_"):
            assert np.allclose(v, fmap["s1_" + n[4:]], equal_nan=True), n
        elif n.startswith("d_"):
            assert np.allclose(v, fmap[n], equal_nan=True), n
    for n, v in zip(s2n, s2v):
        if n.startswith("own_"):
            assert np.allclose(v, fmap["s2_" + n[4:]], equal_nan=True), n
        elif n.startswith("d_"):
            assert np.allclose(v, -fmap[n], equal_nan=True), n

    speed = fmap["speed_mps"]
    print("self-check passed")
    print(f"speed            : {speed:.3f} m/s ({speed * 3.6:.2f} km/h)")
    print(f"per-channel K    : {len(CHANNEL_FEATURE_NAMES)}")
    print(f"file vector      : {len(fv)} features, NaN = {int(np.isnan(fv).sum())}")
    print(f"side vector      : {len(s1v)} features, NaN = {int(np.isnan(s1v).sum())}")
    print(f"load / extract   : {t1 - t0:.3f} s / {t2 - t1:.3f} s")
    print("first names      :", fn[:6])


if __name__ == "__main__":
    _self_check()
