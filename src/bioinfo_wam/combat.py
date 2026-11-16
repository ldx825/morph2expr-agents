"""批次效应校正 —— ComBat 的位置-尺度核心（供智能体 6 使用）。

方法学说明
----------
Johnson, Li & Rabinovic (2007) 的 ComBat 模型为

    y_ij = alpha_j + gamma_ij + delta_ij * eps_ij

其中 alpha_j 是特征 j 的全局均值，gamma_ij / delta_ij 是批次 i 对特征 j 的
位置 / 尺度扰动。校正目标是

    y*_ij = (y_ij - mu_ij) * (sigma_j / sigma_ij) + alpha_j

即：**去掉批次特异的位置与尺度，再还原到全局位置与尺度**。

实现要点（本项目实际踩过的坑，值得写进报告）
--------------------------------------------
1. **不能**写成 `(y - gamma) / sd`。那样会把全局均值也一起除以逐批次的
   sd，等于给每个批次额外引入 `alpha_j / sigma_ij` 的偏移——批次反而变得
   **更**好区分（实测批次分类准确率从 0.94 涨到 1.00）。
   正确做法是先标准化、再乘回全局尺度。
2. **尺度下限**：某些特征在某批次内几乎不变（方差≈0）。若直接除以
   sqrt(方差)，会把数值噪声放大数千倍，分类器可借此完美识别批次。
   因此给 sd 设一个相对于全局尺度的下限（默认 5%）。
3. **经验贝叶斯（EB）收缩**：当"每批样本数 < 特征数"时（本数据即如此：
   586 特征 / 每批约 30–370 孔），EB 会把 gamma/delta 强烈收缩到全局值，
   校正几乎失效。因此默认关闭 EB，退化为标准的**位置-尺度校正**
   （这也是 Cell Painting 领域按板校正的常规做法）。
   打开 EB 可用于演示"收缩过强导致校正不足"。

`strength` 参数在"不校正(0)"与"完整校正(1)"之间线性插值，
用于演示**过校正**风险（校正太狠会把生物学信号一起抹掉）。
"""

from __future__ import annotations

import numpy as np

SD_FLOOR_FRAC = 0.05   # 批次内尺度相对于全局尺度的下限


def _eb_estimates(Y: np.ndarray, batch: np.ndarray, levels: np.ndarray,
                  gamma_hat: np.ndarray, delta_hat: np.ndarray):
    """经验贝叶斯收缩估计（可选），返回收缩后的 gamma / delta。"""
    m = float(gamma_hat.mean())
    t2 = float(gamma_hat.var(ddof=1)) if gamma_hat.shape[0] > 1 else 1.0
    t2 = max(t2, 1e-8)
    a_prior = max(float(delta_hat.mean()), 1e-8)
    b_prior = max(float(delta_hat.var(ddof=1)), 1e-8)

    gamma_star = gamma_hat.copy()
    delta_star = delta_hat.copy()
    for i, lv in enumerate(levels):
        idx = batch == lv
        n = int(idx.sum())
        if n < 2:
            continue
        sub = Y[idx]
        g_old, d_old = gamma_hat[i].copy(), delta_hat[i].copy()
        for _ in range(100):
            g_new = (t2 * n * g_old + d_old * m) / (t2 * n + d_old)
            sum2 = np.sum((sub - g_new[None, :]) ** 2, axis=0)
            d_new = (0.5 * sum2 + b_prior) / (n / 2.0 + a_prior - 1.0)
            converged = (np.max(np.abs(g_new - g_old)) < 1e-4
                         and np.max(np.abs(d_new - d_old)) < 1e-4)
            g_old, d_old = g_new, d_new
            if converged:
                break
        gamma_star[i] = g_old
        delta_star[i] = d_old
    return gamma_star, delta_star


def fit_transform(Y: np.ndarray, batch: np.ndarray,
                  strength: float = 1.0,
                  use_eb: bool = False) -> tuple[np.ndarray, dict]:
    """Y: (n_samples, n_features)；batch: (n_samples,)。

    use_eb=False 时使用**位置－尺度（LS）调整**：直接用每批的均值/方差估计，
    不做经验贝叶斯收缩。当 每批样本数 < 特征数 时，EB 会强烈收缩 gamma/delta，
    导致校正基本失效（本数据集正是这种情况：586 特征 / 每批 ~数百孔）。
    Cell Painting 领域常规做法也是按板做位置-尺度校正，故默认关闭 EB。

    返回校正后的矩阵与诊断信息。
    """
    Y = np.asarray(Y, dtype=np.float64)
    batch = np.asarray(batch).astype(str)
    n, p = Y.shape
    levels = np.unique(batch)
    if len(levels) < 2 or strength <= 0:
        return Y.copy(), {"applied": False, "n_batches": int(len(levels)),
                          "strength": float(strength)}

    grand = Y.mean(axis=0)                        # alpha_j
    sd_global = Y.std(axis=0, ddof=0)
    sd_global = np.where(sd_global > 1e-12, sd_global, 1.0)

    mu = np.zeros((len(levels), p))
    sd = np.ones((len(levels), p))
    n_in = np.zeros(len(levels), dtype=int)

    for i, lv in enumerate(levels):
        idx = batch == lv
        n_in[i] = int(idx.sum())
        sub = Y[idx]
        mu[i] = sub.mean(axis=0)
        s = sub.std(axis=0, ddof=1) if n_in[i] >= 2 else sd_global.copy()
        # 尺度下限：避免把近常数特征的数值噪声放大
        s = np.maximum(s, SD_FLOOR_FRAC * sd_global)
        sd[i] = np.where(s > 1e-12, s, 1.0)

    if use_eb:
        gamma_star, delta_star = _eb_estimates(Y, batch, levels,
                                               mu - grand, sd ** 2)
        mu_star = grand[None, :] + gamma_star
        sd_star = np.sqrt(np.clip(delta_star, 1e-12, None))
        sd_star = np.maximum(sd_star, SD_FLOOR_FRAC * sd_global[None, :])
    else:
        mu_star, sd_star = mu, sd

    # ---- 完整校正：去掉批次位置/尺度，还原到全局位置/尺度 -----------------
    Yc_full = Y.copy()
    for i, lv in enumerate(levels):
        idx = batch == lv
        Yc_full[idx] = (Y[idx] - mu_star[i]) * (sd_global / sd_star[i]) + grand

    # ---- 校正强度插值（1.0 = 完整校正，0.0 = 原样）------------------------
    Yc = (1.0 - strength) * Y + strength * Yc_full

    bad = ~np.isfinite(Yc)
    if bad.any():
        Yc[bad] = Y[bad]

    return Yc, {
        "applied": True,
        "n_batches": int(len(levels)),
        "batch_sizes": {str(l): int(s) for l, s in zip(levels, n_in)},
        "strength": float(strength),
        "use_eb": bool(use_eb),
        "sd_floor_frac": SD_FLOOR_FRAC,
        "mean_abs_location_shift": float(np.abs(mu - grand[None, :]).mean()),
        "mean_scale_ratio": float((sd_star / sd_global[None, :]).mean()),
        "n_nonfinite_fixed": int(bad.sum()),
    }
