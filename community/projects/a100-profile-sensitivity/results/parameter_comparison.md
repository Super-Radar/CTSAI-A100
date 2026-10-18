### 参数对照表（公式与 get_radar_paramete.m 一致）

| 物理量 | 单位 | init0 远波 | init1 近波 |
|---|---|---|---|
| 起始频率 fmcw_startfreq | GHz | 76.3 | 76.119 |
| 扫频带宽 fmcw_bandwidth | MHz | 300 | 750 |
| 斜坡时长 fmcw_chirp_rampup | us | 43 | 104.65 |
| 斜率 S = B/T | MHz/us | 6.97674 | 7.16675 |
| chirp 周期 fmcw_chirp_period | us | 48 | 108.5 |
| chirp 数 nchirp | - | 256 | 128 |
| ADC 采样率 adc_freq | MHz | 25 | 25 |
| 采样窗 adc_sample_start..end | us | 2..42.8 | 2..104.4 |
| adc_sample_num (round+钳位) | 样点 | 1020 (1020, 未钳位) | 2048 (2560→2048) |
| 每 chirp 32bit 字数 rng_nfft/2 | 字 | 512 | 1024 |
| rng_nfft / vel_nfft | - | 1024 / 256 | 2048 / 128 |
| 最大不模糊距离 max_dis | m | 268.75 | 261.625 |
| 距离分辨率 rng_res | m | 0.526961 | 0.204395 |
| RV 图距离 bin rng_res_1dfft | m/bin | 0.524902 | 0.255493 |
| 帧时长 fram_time | ms | 12.288 | 13.888 |
| 不模糊速度跨度 maxV | m/s | 40.9567 | 18.1622 |
| 速度分辨率 V_res = V_acu | m/s | 0.159987 | 0.141892 |
| 每通道期望字数 payload | uint32 | 131072 | 131072 |
