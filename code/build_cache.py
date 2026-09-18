"""程序：数据预处理与缓存。

输入：Train/ 与 Test/ 中的全部原始 CSV，以及 Train_Labels.csv（只读，不修改）。
输出（写入 CACHE_DIR，位于仓库之外）：
  raw_npy/Train/*.npy, raw_npy/Test/*.npy   原始信号的 float32 二进制缓存
  features_train.pkl                        训练集特征表（含标签）
  features_test.pkl                         测试集特征表
每个 .pkl 为字典 {"file": 文件级特征表, "side": 单侧视图特征表}。

用法：
  python code/build_cache.py                 生成原始信号缓存与特征表
  python code/build_cache.py --no-raw-cache  只生成特征表，不写 .npy
已存在的 .npy 会被直接复用；修改 features.py 后重新运行即可更新特征表。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd

import config as C
import data_io
import features

META_COLS = ("filename", "side", "label", "label_id", "side_fault")


def _save_npy_atomic(arr: np.ndarray, path: Path) -> None:
    """先写临时文件再重命名，避免中断时留下不完整的缓存。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(arr, dtype=np.float32))
    os.replace(tmp, path)


def process_file(task: tuple[str, bool]):
    """处理单个文件：读取（优先缓存）→ 必要时写缓存 → 提取特征。"""
    csv_path, write_raw = Path(task[0]), task[1]
    npy_path = data_io.raw_cache_path(csv_path)
    wrote = False
    if npy_path.exists():
        raw = np.load(npy_path, mmap_mode="r")
    else:
        raw = data_io.read_csv_array(csv_path)
        if write_raw:
            _save_npy_atomic(raw, npy_path)
            wrote = True
    pulse, signals = data_io.split_speed_and_signals(raw)
    return csv_path.name, features.extract_all(pulse, signals), wrote


def build_split(files: list[Path], write_raw: bool, workers: int,
                labels: pd.DataFrame | None) -> tuple[dict, int]:
    tasks = [(str(p), write_raw) for p in files]
    results = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, res in enumerate(ex.map(process_file, tasks, chunksize=4), 1):
            results.append(res)
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i:>3}/{len(tasks)}  {time.perf_counter() - t0:6.1f} s", flush=True)

    file_names = results[0][1]["file"][1]
    side_names = results[0][1]["Side I"][1]
    fnames, file_rows, side_rows, n_written = [], [], [], 0
    for fname, out, wrote in results:
        assert out["file"][1] == file_names, f"{fname}: 文件级特征名不一致"
        assert out["Side I"][1] == side_names and out["Side II"][1] == side_names, \
            f"{fname}: 单侧视图特征名不一致"
        fnames.append(fname)
        file_rows.append(out["file"][0])
        side_rows.extend(out[side][0] for side in C.SIDES)
        n_written += int(wrote)

    df_file = pd.DataFrame(np.vstack(file_rows), columns=file_names)
    df_file.insert(0, "filename", fnames)

    df_side = pd.DataFrame(np.vstack(side_rows), columns=side_names)
    df_side.insert(0, "side", [s for _ in fnames for s in C.SIDES])
    df_side.insert(0, "filename", [f for f in fnames for _ in C.SIDES])

    if labels is not None:
        lab = labels.set_index(C.LABELS_FILE_COL)
        df_file.insert(1, "label", lab.loc[fnames, C.LABELS_LABEL_COL].to_numpy())
        df_file.insert(2, "label_id", lab.loc[fnames, "label_id"].to_numpy())
        file_label = df_side["filename"].map(lab[C.LABELS_LABEL_COL])
        df_side.insert(2, "label", file_label.to_numpy())
        df_side.insert(3, "side_fault", (file_label == df_side["side"]).astype(int).to_numpy())

    return {"file": df_file, "side": df_side}, n_written


def summarize(tables: dict, n_written: int, elapsed: float) -> None:
    f, s = tables["file"], tables["side"]
    fcols = [c for c in f.columns if c not in META_COLS]
    x = f[fcols].to_numpy(dtype=np.float64)
    nan_cols = [c for c in fcols if f[c].isna().any()]
    sp = f["speed_mps"]
    print(f"  file table     : {f.shape[0]} rows x {len(fcols)} features")
    print(f"  side table     : {s.shape[0]} rows x {s.shape[1] - sum(c in META_COLS for c in s.columns)} features")
    print(f"  NaN / inf      : {int(np.isnan(x).sum())} / {int(np.isinf(x).sum())}"
          f"  (columns with NaN: {len(nan_cols)})")
    print(f"  speed m/s      : min {sp.min():.3f}  median {sp.median():.3f}  max {sp.max():.3f}")
    print(f"  speed < {C.MIN_SPEED_MPS} m/s : {int((sp < C.MIN_SPEED_MPS).sum())} files")
    print(f"  npy written    : {n_written}")
    print(f"  elapsed        : {elapsed:.1f} s")
    if "label" in f.columns:
        print("  file labels    :", f["label"].value_counts().to_dict())
        print("  side_fault     :", s["side_fault"].value_counts().to_dict())


def main() -> None:
    ap = argparse.ArgumentParser(description="Rail Corrugation 数据预处理与缓存")
    ap.add_argument("--no-raw-cache", action="store_true", help="只生成特征表，不写 .npy")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="并行进程数")
    args = ap.parse_args()

    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    labels = data_io.load_labels()
    train_files = data_io.list_csv_files(C.TRAIN_DIR)
    test_files = data_io.list_csv_files(C.TEST_DIR)
    assert [p.name for p in train_files] == labels[C.LABELS_FILE_COL].tolist()

    for name, files, lab, out_path in [
        ("train", train_files, labels, C.FEATURES_TRAIN),
        ("test", test_files, None, C.FEATURES_TEST),
    ]:
        print(f"[{name}] {len(files)} files, workers = {args.workers}")
        t0 = time.perf_counter()
        tables, n_written = build_split(files, not args.no_raw_cache, args.workers, lab)
        pd.to_pickle(tables, out_path)
        summarize(tables, n_written, time.perf_counter() - t0)
        print(f"  saved -> {out_path.relative_to(C.WORKSPACE_ROOT)}")


if __name__ == "__main__":
    main()
