"""特征提取（纯函数，不涉及文件读写）。

分两个层次：
1. 逐轴箱特征：extract_channel_features(pulse, signals) -> (车速, (8, 8, 2, K))
2. 聚合特征：file_level / side_level，输入一批逐轴箱特征 (N, 8, 8, 2, K)，
   可选传入传感器基线 baseline (8, 8, 2, K)，先减基线再跨轴箱聚合。
基线由 fit_baseline 在训练数据上估计；须在交叉验证每一折内部拟合，避免信息泄漏。
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy import signal as sps
from scipy import stats

import config as C

EPS = 1e-12
CH_TAGS = ("vib", "shk")                         # 与 C.CHANNEL_TYPES 顺序对应
_SIDE_I = list(C.SIDE_I_POS_IDX)
_SIDE_II = list(C.SIDE_II_POS_IDX)
_PAIR_I = [a - 1 for a, _ in C.AXLE_PAIRS]
_PAIR_II = [b - 1 for _, b in C.AXLE_PAIRS]


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
K = len(CHANNEL_FEATURE_NAMES)


# ============================================================ 第一层：逐轴箱特征
def compute_speed_mps(pulse: np.ndarray) -> float:
    """由测速齿轮脉冲的 0/1 跳变次数计算车速（m/s）。"""
    n_transitions = int(np.count_nonzero(np.diff(np.asarray(pulse))))
    duration_s = len(pulse) / C.FS_HZ
    revs_per_s = n_transitions / C.TRANSITIONS_PER_REV / duration_s
    return float(revs_per_s * C.WHEEL_CIRCUMFERENCE_M)


def channel_features(signals: np.ndarray, speed_mps: float) -> np.ndarray:
    """对每一路信号计算特征，返回 (8, 8, 2, K)。"""
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


def extract_channel_features(pulse: np.ndarray, signals: np.ndarray) -> tuple[float, np.ndarray]:
    """单个文件：返回 (车速 m/s, 逐轴箱特征 (8, 8, 2, K) float32)。"""
    v = compute_speed_mps(pulse)
    return v, channel_features(signals, v).astype(np.float32)


# ============================================================ 第二层：基线与聚合
def fit_baseline(cf: np.ndarray) -> np.ndarray:
    """各传感器、各特征在一批文件上的中位数，返回 (8, 8, 2, K)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(np.asarray(cf, dtype=np.float64), axis=0)


def _reduce(x: np.ndarray, stat: str) -> np.ndarray:
    """沿轴 1（轴箱或轮对）计算统计量并忽略 NaN：(N, M, 2, K) -> (N, 2, K)。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if stat == "mean":
            return np.nanmean(x, axis=1)
        if stat == "median":
            return np.nanmedian(x, axis=1)
        if stat == "max":
            return np.nanmax(x, axis=1)
        if stat == "min":
            return np.nanmin(x, axis=1)
        if stat == "std":
            return np.nanstd(x, axis=1)
        if stat.startswith("q"):
            return np.nanpercentile(x, float(stat[1:]), axis=1)
    raise ValueError(f"未知统计量: {stat}")


def _prepare(cf: np.ndarray, baseline: np.ndarray | None):
    """减基线后拆分为 Side I (N,32,2,K)、Side II (N,32,2,K) 与配对差值 (N,32,2,K)。"""
    x = np.asarray(cf, dtype=np.float64)
    if x.ndim == 4:
        x = x[None]
    if baseline is not None:
        x = x - np.asarray(baseline, dtype=np.float64)
    n = x.shape[0]
    shape = (n, -1, C.N_CHANNEL_TYPES, x.shape[-1])
    s1 = x[:, :, _SIDE_I].reshape(shape)
    s2 = x[:, :, _SIDE_II].reshape(shape)
    pdiff = (x[:, :, _PAIR_I] - x[:, :, _PAIR_II]).reshape(shape)
    return s1, s2, pdiff


def _block(arr: np.ndarray, prefix: str, stat: str) -> tuple[np.ndarray, list[str]]:
    names = [f"{prefix}_{ch}_{fn}_{stat}" for ch in CH_TAGS for fn in CHANNEL_FEATURE_NAMES]
    return arr.reshape(arr.shape[0], -1), names


def _side_stats(arr: np.ndarray) -> dict[str, np.ndarray]:
    need = tuple(dict.fromkeys(C.AGG_STATS + C.CONTRAST_STATS))
    return {s: _reduce(arr, s) for s in need}


def file_level(speeds, cf: np.ndarray, baseline: np.ndarray | None = None
               ) -> tuple[np.ndarray, list[str]]:
    """文件级特征：车速 + 两侧同侧聚合 + 两侧之差 + 配对差值聚合。返回 (N, D), 特征名。"""
    s1, s2, pdiff = _prepare(cf, baseline)
    speeds = np.atleast_1d(np.asarray(speeds, dtype=np.float64))
    st1, st2 = _side_stats(s1), _side_stats(s2)

    cols, names = [speeds[:, None]], ["speed_mps"]

    def add(block):
        cols.append(block[0])
        names.extend(block[1])

    for tag, st in (("s1", st1), ("s2", st2)):
        for s in C.AGG_STATS:
            add(_block(st[s], tag, s))
    for s in C.CONTRAST_STATS:
        add(_block(st1[s] - st2[s], "d", s))
    for s in C.PAIR_STATS:
        add(_block(_reduce(pdiff, s), "p", s))
    return np.hstack(cols), names


def side_level(speeds, cf: np.ndarray, baseline: np.ndarray | None = None
               ) -> tuple[np.ndarray, list[str], np.ndarray, np.ndarray]:
    """单侧视图特征：每个文件拆为 Side I、Side II 两行。

    返回 X (2N, D)、特征名、所属文件下标 (2N,)、侧别 (2N,)。
    配对差值统一为“本侧 - 对侧”方向。
    """
    s1, s2, pdiff = _prepare(cf, baseline)
    speeds = np.atleast_1d(np.asarray(speeds, dtype=np.float64))
    n = speeds.shape[0]
    st1, st2 = _side_stats(s1), _side_stats(s2)

    views, names = [], []
    for side, own, other, sign in (("Side I", st1, st2, 1.0), ("Side II", st2, st1, -1.0)):
        cols = [speeds[:, None], np.full((n, 1), float(side == "Side II"))]
        names = ["speed_mps", "is_side_II"]

        def add(block):
            cols.append(block[0])
            names.extend(block[1])

        for s in C.AGG_STATS:
            add(_block(own[s], "own", s))
        for s in C.CONTRAST_STATS:
            add(_block(own[s] - other[s], "d", s))
        for s in C.PAIR_STATS:
            add(_block(_reduce(sign * pdiff, s), "p", s))
        views.append(np.hstack(cols))

    x = np.stack(views, axis=1).reshape(2 * n, -1)
    file_idx = np.repeat(np.arange(n), 2)
    sides = np.array(list(C.SIDES) * n)
    return x, names, file_idx, sides


# ============================================================ 自检
def _mirror_stat(stat: str) -> str | None:
    if stat == "max":
        return "min"
    if stat == "min":
        return "max"
    if stat.startswith("q"):
        return f"q{100 - int(stat[1:])}"
    return None


def _self_check() -> None:
    import time
    import data_io

    files = [C.TRAIN_DIR / "Train1.csv", C.TRAIN_DIR / "Train2.csv"]
    t0 = time.perf_counter()
    res = [extract_channel_features(*data_io.load_file(p)) for p in files]
    t1 = time.perf_counter()
    speeds = np.array([r[0] for r in res])
    cf = np.stack([r[1] for r in res])
    assert cf.shape == (2, C.N_CARS, C.N_POSITIONS, C.N_CHANNEL_TYPES, K)

    for label, baseline in (("no baseline", None), ("baseline", fit_baseline(cf))):
        fx, fn = file_level(speeds, cf, baseline)
        sx, sn, fidx, sides = side_level(speeds, cf, baseline)
        assert fx.shape == (2, len(fn)) and len(set(fn)) == len(fn)
        assert sx.shape == (4, len(sn)) and len(set(sn)) == len(sn)
        assert not np.isinf(fx).any() and not np.isinf(sx).any()

        fmaps = [dict(zip(fn, row)) for row in fx]
        for r in range(sx.shape[0]):
            fm, side = fmaps[fidx[r]], sides[r]
            tag, sign = ("s1", 1.0) if side == "Side I" else ("s2", -1.0)
            for name, v in zip(sn, sx[r]):
                if name.startswith("own_"):
                    exp = fm[f"{tag}_" + name[4:]]
                elif name.startswith("d_"):
                    exp = sign * fm[name]
                elif name.startswith("p_"):
                    if side == "Side I":
                        exp = fm[name]
                    else:
                        base, stat = name.rsplit("_", 1)
                        mirror = _mirror_stat(stat)
                        key = f"{base}_{mirror}" if mirror else None
                        if key not in fm:
                            continue
                        exp = -fm[key]
                else:
                    continue
                assert np.allclose(v, exp, equal_nan=True), (label, side, name)
        print(f"[{label}] file {fx.shape}, side {sx.shape}, "
              f"NaN file/side = {int(np.isnan(fx).sum())}/{int(np.isnan(sx).sum())}")

    print("self-check passed")
    print(f"speeds (m/s)     : {np.round(speeds, 3).tolist()}")
    print(f"per-channel K    : {K}")
    print(f"extract 2 files  : {t1 - t0:.3f} s")


if __name__ == "__main__":
    _self_check()
