# -*- coding: utf-8 -*-
"""真实数据冒烟测试（仓库内示例数据存在时才运行）。"""
import os

import numpy as np
import pytest

from rx_ablation.pipeline import (
    DEFAULT_CFG_DIR,
    DEFAULT_DATA_DIR,
    RadarConfig,
    load_dataset,
    repo_root_from_module,
)

REPO = repo_root_from_module()
HAS_DATA = os.path.isdir(os.path.join(REPO, DEFAULT_DATA_DIR))

pytestmark = pytest.mark.skipif(
    not HAS_DATA, reason="仓库内无示例 ADC 数据（社区裁剪仓库时跳过）"
)


def test_far_frame_shape_finite_and_dominant_range():
    cfg = RadarConfig.from_hxx(
        os.path.join(REPO, DEFAULT_CFG_DIR, "sensor_config_init0.hxx")
    )
    frame = load_dataset(os.path.join(REPO, DEFAULT_DATA_DIR), cfg, "f0")
    assert frame["power"].shape == (512, 256, 4)
    assert frame["rv_image"].shape == (512, 256)
    assert np.isfinite(frame["power"]).all()
    db = 10 * np.log10(frame["rv_image"])
    med = float(np.median(db))
    # 交叉验证锚点 1：社区 a100-matlab-adc-pipeline-validation 在同一批数据上
    # 检出 6.29 m（0 基 bin≈12）/ Doppler 0 目标，报告 128.46 dB —— 该项目
    # 未除官方 FFT scalar（10log10(255^2*63^2)≈84.1 dB），换算后与本链路一致。
    assert db[12, 0] - med > 40, "6.29 m 目标（bin 12, vel 0）缺失"
    # 交叉验证锚点 2：官方主脚本在 targetRange=2 m ±10 bin 搜索窗内的最强
    # 峰位于近距 bin 5（2.6 m）/ vel 0，是全图最大值。
    pk = np.unravel_index(np.argmax(frame["rv_image"]), frame["rv_image"].shape)
    assert pk[1] == 0, f"最强峰 Doppler bin 异常: {pk}"
    assert 3 <= pk[0] <= 7, f"最强峰距离 bin 异常: {pk}"
    assert db[5, 0] - med > 45


def test_near_frame_shape():
    cfg = RadarConfig.from_hxx(
        os.path.join(REPO, DEFAULT_CFG_DIR, "sensor_config_init1.hxx")
    )
    frame = load_dataset(os.path.join(REPO, DEFAULT_DATA_DIR), cfg, "f1")
    assert frame["power"].shape == (1024, 128, 4)
    assert cfg.adc_sample_num == 2048  # 2560 被 rng_nfft 截断
