# 运行摘要（cli 一键复现输出）

- 种子: 20260929; 数据目录: `/home/cnj/openSource/wt-profile-sensitivity/ADC数据采集/示例adc数据和结果`
- 载荷陷阱: init0 与 init1 每通道期望字数均为 131072（长度检查无法区分）
- init0: max_dis=268.750 m, rng_res=0.5270 m, maxV=40.957 m/s, fram=12.288 ms
- init1: max_dis=261.625 m, rng_res=0.2044 m, maxV=18.162 m/s, fram=13.888 ms
- 距离轴误配比例: 近波→按init0 ×1.02723, 远波→按init1 ×0.97349
- 速度轴误配比例: 近波→按init0 ×1.12753, 远波→按init1 ×0.88690
- 合成对拍最大误差: |ΔR|≤0.482 bin, |Δv|≤0.499 bin
- 半帧速度跳变: 1/4 组合发生 (近波按 init0 展开时, chirp 从中点切开形成 period-2 相位调制, 谱峰弹到 ±maxV/2)
- 真实数据交叉表: 2/8 行谱峰可按比例预测, 6 行 peak_jump（主峰跳到泄漏格，已在表中标注）
- 检测器: 公开 8 文件 8/8 判对；空场景合成远波仍判对；纯噪声合成流 abstain（uncertain）
