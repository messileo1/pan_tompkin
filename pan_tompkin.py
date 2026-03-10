"""
@Project ：
@File    ：pan_tompkin_optimized.py
@Description: 经典的pan-tompinks算法 —— numpy加速优化版本
"""

# !/usr/bin/env python
# -*- coding: UTF-8 -*-

from scipy import signal
import numpy as np
import data.read_data as rd
import wfdb
import os
from utils import ecg_display as display


def bandpass_filter(data, fs, low=5, high=15):
    """
    带通滤波
    :param data: array-like, 输入信号数据
    :param fs: int, 采样率
    :param low: int, 截止频率1
    :param high: int, 截止频率2
    :return: np.ndarray, 滤波后信号
    """
    low = 2 * low / fs
    high = 2 * high / fs
    b, a = signal.butter(3, [low, high], 'bandpass')
    return signal.filtfilt(b, a, data)


# ──────────────────────────────────────────────
# 核心信号处理：全部换成 numpy，避免 Python 循环
# ──────────────────────────────────────────────

def derivative(data):
    """
    求导函数 H(z) = (1/8T)(-z^{-2} - 2z^{-1} + 2z + z^{2})
    原实现：手写双层循环 conv → 替换为 np.convolve，速度提升 ~100x
    """
    coef = np.array([-1, -2, 0, 2, 1], dtype=float)
    # 'same' 保持与原始数据等长
    return np.convolve(np.asarray(data, dtype=float), coef, mode='same')


def square(data):
    """平方：列表推导 → np.square"""
    return np.square(np.asarray(data, dtype=float))


def moving_window_average(data, fs=250):
    """
    移动窗口积分均值
    原实现：手写 conv 循环 → np.convolve 均值滤波
    """
    win_width = int(0.15 * fs + 0.5)
    kernel = np.ones(win_width, dtype=float) / win_width
    return np.convolve(np.asarray(data, dtype=float), kernel, mode='same')


def findpeaks(data, min_distance, fs=250):
    """
    寻找峰值位置 peak，返回峰值和位置
    原实现：手写状态机循环 → scipy.signal.find_peaks，速度提升显著
    :param data: array-like
    :param min_distance: 两峰最小距离（样本数）
    :param fs: 采样率
    :return: peaks(np.ndarray), locs(np.ndarray)
    """
    min_dist = int(0.1 * fs + 0.5)
    data = np.asarray(data, dtype=float)
    locs, props = signal.find_peaks(data, distance=min_dist)
    peaks = data[locs]
    return peaks, locs


def find_max(vector):
    """
    找最大值及其位置
    原实现：手写循环 → np.argmax
    """
    arr = np.asarray(vector, dtype=float)
    if arr.size == 0:
        return 0, 0
    max_i = int(np.argmax(arr))
    return arr[max_i], max_i


def _mean(data):
    """mean 封装，兼容空列表"""
    arr = np.asarray(data, dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _diff(data):
    """diff 封装，保持与原 diff() 相同行为（在第0位插入0）"""
    arr = np.asarray(data, dtype=float)
    if arr.size < 2:
        return np.zeros(len(arr))
    return np.concatenate(([0.0], np.diff(arr)))


def judge_rule(ecg_filter, ecg_win, peaks, locs, fs=250):
    """
    自适应阈值处理（主检测逻辑）
    内部循环依赖状态机，难以完全向量化；
    但将所有内层的列表操作换成 numpy 切片/运算，减少重复创建对象的开销。
    :param ecg_filter: np.ndarray, 带通滤波后的数据
    :param ecg_win:    np.ndarray, 积分窗后的数据
    :param peaks:      np.ndarray, 通过积分窗后找到的峰值
    :param locs:       np.ndarray, peaks 对应的下标
    :param fs: int, 采样率
    :return: (qrs_amp_win, qrs_idx_win, qrs_amp_flt, qrs_idx_flt,
              thrs_win1, thrs_win2, thrs_flt1, thrs_flt2)
    """
    ecg_filter = np.asarray(ecg_filter, dtype=float)
    ecg_win    = np.asarray(ecg_win,    dtype=float)
    peaks      = np.asarray(peaks,      dtype=float)
    locs       = np.asarray(locs,       dtype=int)

    n = len(peaks)
    N_flt = len(ecg_filter)
    N_win = len(ecg_win)

    # 预分配输出（用 list append，最后转 np.array）
    qrs_amp_win, qrs_idx_win = [], []
    qrs_amp_flt, qrs_idx_flt = [], []
    thrs_win1, thrs_win2     = [], []
    thrs_flt1, thrs_flt2     = [], []

    # 初始化阈值（前 2 秒）
    init_win = ecg_win[:2 * fs]
    init_flt = ecg_filter[:2 * fs]
    THRESHOLD_I1 = 0.25 * float(np.max(init_win))
    THRESHOLD_I2 = 0.5  * float(np.mean(init_win))
    SPKI, NPKI   = THRESHOLD_I1, THRESHOLD_I2

    THRESHOLD_F1 = 0.25 * float(np.max(init_flt))
    THRESHOLD_F2 = 0.5  * float(np.mean(init_flt))
    SPKF, NPKF   = THRESHOLD_F1, THRESHOLD_F2

    RR_AVERAGE2 = 0.0
    rr_recent_limit = np.zeros(8, dtype=float)
    rr_limit_idx, rr_limit_count = 0, 0

    is_t_wave    = False
    is_first_win = False
    is_new_qrs   = False

    delay = int(0.15 * fs + 0.5)

    for i in range(n):
        PEAKI     = float(peaks[i])
        PEAKI_IDX = int(locs[i])

        # ── 在带通滤波数据中定位峰值 ──────────────────────────
        PEAKF, PEAKF_IDX = 0, 0
        lo = PEAKI_IDX - delay
        if lo >= 1 and PEAKI_IDX <= N_flt:
            PEAKF, PEAKF_IDX = find_max(ecg_filter[lo:PEAKI_IDX])
        elif i == 0:
            PEAKF, PEAKF_IDX = find_max(ecg_filter[:PEAKI_IDX])
            is_first_win = True
        elif PEAKI_IDX >= N_flt:
            PEAKF, PEAKF_IDX = find_max(ecg_filter[lo:])

        # ── 更新心率 ──────────────────────────────────────────
        if len(qrs_idx_win) >= 9:
            rr_arr    = np.diff(qrs_idx_win[-9:])  # 快速求差
            RR_AVERAGE1 = float(np.mean(rr_arr))
            latest_rr   = qrs_idx_win[-1] - qrs_idx_win[-2]
            if rr_limit_count == 0:
                RR_AVERAGE2 = RR_AVERAGE1
            elif rr_limit_count < 8:
                RR_AVERAGE2 = float(np.mean(rr_recent_limit[:rr_limit_count]))
            else:
                RR_AVERAGE2 = float(np.mean(rr_recent_limit))

            if latest_rr <= 0.92 * RR_AVERAGE2 or latest_rr >= 1.16 * RR_AVERAGE2:
                THRESHOLD_I1 *= 0.5
                THRESHOLD_F1 *= 0.5
            elif is_new_qrs:
                rr_recent_limit[rr_limit_idx] = latest_rr
                rr_limit_idx = (rr_limit_idx + 1) % 8
                rr_limit_count += 1
                is_new_qrs = False

        # ── 回找策略 ──────────────────────────────────────────
        if RR_AVERAGE2 != 0 and qrs_idx_win:
            gap = int(1.66 * RR_AVERAGE2 + 0.5)
            margin = int(0.2 * fs + 0.5)
            if PEAKI_IDX - qrs_idx_win[-1] >= gap:
                sb = qrs_idx_win[-1] + margin
                eb = PEAKI_IDX - margin
                if sb < eb:
                    PEAKI, PEAKI_IDX_tmp = find_max(ecg_win[sb:eb])
                    PEAKI_IDX_tmp = sb + PEAKI_IDX_tmp - 1  # 转绝对坐标
                    if PEAKI > THRESHOLD_I2:
                        qrs_amp_win.append(PEAKI)
                        qrs_idx_win.append(PEAKI_IDX_tmp)
                        SPKI = 0.25 * PEAKI + 0.75 * SPKI
                        is_new_qrs = True
                        lo2 = PEAKI_IDX_tmp - delay
                        if PEAKI_IDX_tmp <= N_flt:
                            find_range = ecg_filter[lo2:PEAKI_IDX_tmp]
                        else:
                            find_range = ecg_filter[lo2:]
                        PEAKF_B, PEAKF_B_IDX = find_max(find_range)
                        if PEAKF_B > THRESHOLD_F2:
                            qrs_amp_flt.append(PEAKF_B)
                            qrs_idx_flt.append(lo2 + PEAKF_B_IDX)
                            SPKF = 0.25 * PEAKF_B + 0.75 * SPKF

        # ── 阈值判断 ──────────────────────────────────────────
        if peaks[i] > THRESHOLD_I1:
            if len(qrs_idx_win) >= 3:
                gap2 = PEAKI_IDX - qrs_idx_win[-1]
                if int(0.20 * fs + 0.5) < gap2 <= int(0.36 * fs + 0.5):
                    w = int(0.075 * fs + 0.5)
                    cur_slope = float(np.mean(np.diff(ecg_win[PEAKI_IDX - w:PEAKI_IDX])))
                    pre_slope = float(np.mean(np.diff(ecg_win[qrs_idx_win[-1] - w:qrs_idx_win[-1]])))
                    if abs(cur_slope) < 0.5 * abs(pre_slope):
                        is_t_wave = True
                        NPKF = 0.125 * PEAKF + 0.875 * NPKF
                        NPKI = 0.125 * PEAKI + 0.875 * NPKI
                    else:
                        is_t_wave = False

            if not is_t_wave:
                qrs_amp_win.append(PEAKI)
                qrs_idx_win.append(PEAKI_IDX)
                is_new_qrs = True
                SPKI = 0.125 * PEAKI + 0.875 * SPKI
                if PEAKF >= THRESHOLD_F2:
                    if is_first_win:
                        qrs_idx_flt.append(PEAKF_IDX)
                    else:
                        qrs_idx_flt.append(PEAKI_IDX - delay + PEAKF_IDX)
                    qrs_amp_flt.append(PEAKF)
                    SPKF = 0.125 * PEAKF + 0.875 * SPKF

        elif THRESHOLD_I2 <= peaks[i] < THRESHOLD_I1:
            NPKF = 0.125 * PEAKF + 0.875 * SPKF
            NPKI = 0.125 * PEAKI + 0.875 * SPKI
        else:
            NPKF = 0.125 * PEAKF + 0.875 * SPKF
            NPKI = 0.125 * PEAKI + 0.875 * SPKI

        # ── 自适应阈值更新 ────────────────────────────────────
        if NPKI != 0 or SPKI != 0:
            THRESHOLD_I1 = NPKI + 0.25 * abs(SPKI - NPKI)
            THRESHOLD_I2 = 0.5 * THRESHOLD_I1
        if NPKF != 0 or SPKF != 0:
            THRESHOLD_F1 = NPKF + 0.25 * abs(SPKF - NPKF)
            THRESHOLD_F2 = 0.5 * THRESHOLD_F1

        thrs_win1.append(THRESHOLD_I1)
        thrs_win2.append(THRESHOLD_I2)
        thrs_flt1.append(THRESHOLD_F1)
        thrs_flt2.append(THRESHOLD_F2)

        is_t_wave    = False
        is_first_win = False

    return (qrs_amp_win, qrs_idx_win, qrs_amp_flt, qrs_idx_flt,
            thrs_win1,   thrs_win2,   thrs_flt1,   thrs_flt2)


def get_annotation_count(path, dat_name):
    record_name = os.path.join(path, dat_name.split('.')[0])
    annotation = wfdb.rdann(record_name, 'atr')
    beat_symbols = ['N', 'L', 'R', 'B', 'A', 'a', 'J', 'S', 'V', 'r',
                    'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q', '?']
    beats = [s for s in annotation.symbol if s in beat_symbols]
    print(f"记录 {dat_name} 的详细统计:")
    print(f"   实际心跳总数 (Beats): {len(beats)}")
    return len(beats)


if __name__ == '__main__':
    path = '../../data/mit-bih-arrhythmia'
    dat_name = '100.dat'
    fs = 360

    data_list = [
        100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
        111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
        122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
        209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
        222, 223, 228, 230, 231, 232, 233, 234
    ]

    for rec in data_list:
        dat_name = f'{rec}.dat'

        ecg_data1, ecg_data2 = rd.read_f212(path, dat_name)
        ecg = np.asarray(ecg_data1, dtype=float)  # 转 numpy，后续全程 numpy

        ecg1 = bandpass_filter(ecg, fs, 5, 15)          # 带通滤波
        ecg2 = derivative(ecg1)                          # 求导（np.convolve）
        ecg3 = square(ecg2)                              # 平方（np.square）
        ecg4 = moving_window_average(ecg3, fs)           # 移动窗口均值（np.convolve）
        peaks, locs = findpeaks(ecg4, int(0.2 * fs + 0.5), fs)  # 峰值检测（scipy）

        (qrs_amp_win, qrs_idx_win,
         qrs_amp_flt, qrs_idx_flt,
         thrs_win1, thrs_win2,
         thrs_flt1, thrs_flt2) = judge_rule(ecg1, ecg4, peaks, locs, fs)

        print(f"检测到的 R 波总数: {len(qrs_idx_flt)}")
        all_beats = get_annotation_count(path, dat_name)
        print(f"召回率为: {len(qrs_idx_flt) / all_beats:.4f}")

    print('----------------完成统计----------------')