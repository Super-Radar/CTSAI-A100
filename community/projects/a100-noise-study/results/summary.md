# a100-noise-study 运行结果摘要（自动生成）

- 数据目录：`/home/cnj/openSource/wt-noise-study/ADC数据采集/示例adc数据和结果`
- 随机种子：20260929；bootstrap 次数：2000
- 诚实性声明：公开示例数据无空场真值，本文一切数值均为“相对官方方法/相对无注入基线的偏差”，不构成绝对底噪/SNR 断言。

## 1. 配置与数据装载
### far wave (init0, f0)（sensor_config_init0.hxx）

| 参数 | 值 |
|---|---|
| fmcw_startfreq_GHz | 76.300 |
| fmcw_bandwidth_MHz | 300 |
| fmcw_chirp_rampup_us | 43 |
| fmcw_chirp_period_us | 48 |
| nchirp | 256 |
| adc_freq_MHz | 25 |
| dec_factor | 1 |
| adc_sample_start_us | 2 |
| adc_sample_end_us | 42.800 |
| adc_sample_num | 1020 |
| rng_nfft | 1024 |
| vel_nfft | 256 |
| rng_fft_scalar | 255 |
| vel_fft_scalar | 63 |
| rng_win | 2 |
| vel_win | 2 |
| nvirtual_chirp | 1 |
| nvirtual_array | 4 |

官方噪声区域（MATLAB 1-based 含端点）：range 384..410，速度半区 6..122 与 134..250，共 3159 个 cell（min 合成后）。

| 文件 | 实际值数 | 期望值数 | 处理 | 被丢弃尾值 |
|---|---|---|---|---|
| adc_远波_20260512135827_f0_0_13_58_36_748.txt | 131073 | 131072 | truncate | [0] |
| adc_远波_20260512135827_f0_1_13_58_45_367.txt | 131073 | 131072 | truncate | [0] |
| adc_远波_20260512135827_f0_2_13_58_53_977.txt | 131073 | 131072 | truncate | [0] |
| adc_远波_20260512135827_f0_3_13_59_02_596.txt | 131073 | 131072 | truncate | [0] |

### near wave (init1, f1)（sensor_config_init1.hxx）

| 参数 | 值 |
|---|---|
| fmcw_startfreq_GHz | 76.119 |
| fmcw_bandwidth_MHz | 750 |
| fmcw_chirp_rampup_us | 104.650 |
| fmcw_chirp_period_us | 108.500 |
| nchirp | 128 |
| adc_freq_MHz | 25 |
| dec_factor | 1 |
| adc_sample_start_us | 2 |
| adc_sample_end_us | 104.400 |
| adc_sample_num | 2048 |
| rng_nfft | 2048 |
| vel_nfft | 128 |
| rng_fft_scalar | 255 |
| vel_fft_scalar | 63 |
| rng_win | 2 |
| vel_win | 2 |
| nvirtual_chirp | 1 |
| nvirtual_array | 4 |

官方噪声区域（MATLAB 1-based 含端点）：range 768..819，速度半区 6..58 与 70..122，共 2756 个 cell（min 合成后）。

| 文件 | 实际值数 | 期望值数 | 处理 | 被丢弃尾值 |
|---|---|---|---|---|
| adc_近波_20260512141857_f1_0_14_19_05_965.txt | 131073 | 131072 | truncate | [0] |
| adc_近波_20260512141857_f1_1_14_19_14_570.txt | 131073 | 131072 | truncate | [0] |
| adc_近波_20260512141857_f1_2_14_19_23_170.txt | 131073 | 131072 | truncate | [0] |
| adc_近波_20260512141857_f1_3_14_19_31_788.txt | 131073 | 131072 | truncate | [0] |

## 2. 复现表（官方 noiseValue + 稳健估计量，dB）

| dataset | channel_set | cells | official_mean_dB | boot95(official) | median_dB | boot95(median) |
|---|---|---|---|---|---|---|
| far | ch0 | 3159 | -14.180 | [-14.38, -13.99] | -13.215 | [-13.39, -13.03] |
| far | ch1 | 3159 | -9.688 | [-9.88, -9.49] | -8.849 | [-9.06, -8.62] |
| far | ch2 | 3159 | -9.765 | [-9.95, -9.58] | -8.959 | [-9.19, -8.74] |
| far | ch3 | 3159 | -10.182 | [-10.38, -9.99] | -9.248 | [-9.46, -9.10] |
| far | 4ch_avg_official | 3159 | -7.514 | [-7.59, -7.44] | -7.380 | [-7.50, -7.27] |
| near | ch0 | 2756 | -13.881 | [-14.10, -13.67] | -12.925 | [-13.25, -12.68] |
| near | ch1 | 2756 | -8.753 | [-8.97, -8.55] | -7.818 | [-8.08, -7.61] |
| near | ch2 | 2756 | -8.815 | [-9.02, -8.60] | -7.970 | [-8.21, -7.80] |
| near | ch3 | 2756 | -9.089 | [-9.30, -8.89] | -8.097 | [-8.34, -7.95] |
| near | 4ch_avg_official | 2756 | -6.695 | [-6.79, -6.60] | -6.574 | [-6.67, -6.46] |

## 3. 合成注入污染演示（远波 4 通道，已标注为合成注入）

| scenario | peak_amp_dB | sigma_bins | n_hit_cells | estimator | value_dB | deviation_dB |
|---|---|---|---|---|---|---|
| clean | 0.000 | 0.000 | 0 | official_mean | -7.51 | 0.00 |
| clean | 0.000 | 0.000 | 0 | median | -7.38 | 0.00 |
| clean | 0.000 | 0.000 | 0 | mad_floor | -5.03 | 0.00 |
| single_half_20dB_s2 | 20.000 | 2.000 | 5 | official_mean | -7.49 | 0.03 |
| single_half_20dB_s2 | 20.000 | 2.000 | 5 | median | -7.35 | 0.03 |
| single_half_20dB_s2 | 20.000 | 2.000 | 5 | mad_floor | -5.00 | 0.03 |
| both_halves_20dB_s2 | 20.000 | 2.000 | 45 | official_mean | -7.35 | 0.16 |
| both_halves_20dB_s2 | 20.000 | 2.000 | 45 | median | -7.29 | 0.09 |
| both_halves_20dB_s2 | 20.000 | 2.000 | 45 | mad_floor | -4.89 | 0.14 |
| both_halves_40dB_s2 | 40.000 | 2.000 | 69 | official_mean | -7.20 | 0.32 |
| both_halves_40dB_s2 | 40.000 | 2.000 | 69 | median | -7.27 | 0.11 |
| both_halves_40dB_s2 | 40.000 | 2.000 | 69 | mad_floor | -4.87 | 0.17 |
| both_halves_40dB_s6 | 40.000 | 6.000 | 593 | official_mean | -4.72 | 2.79 |
| both_halves_40dB_s6 | 40.000 | 6.000 | 593 | median | -6.56 | 0.82 |
| both_halves_40dB_s6 | 40.000 | 6.000 | 593 | mad_floor | -3.51 | 1.52 |

## 4. 区域扫描散布

| dataset | vel_mode | n_regions | min_dB | max_dB | spread(max-min)_dB | std_dB |
|---|---|---|---|---|---|---|
| far | two_half_min | 124 | -8.647 | -7.005 | 1.642 | 0.269 |
| far | first_half | 124 | -7.161 | -5.463 | 1.698 | 0.280 |
| near | two_half_min | 141 | -8.249 | -5.681 | 2.568 | 0.448 |
| near | first_half | 141 | -6.606 | -4.186 | 2.419 | 0.419 |

## 5. 偏差曲线关键点（both-halves 注入，合成）

| dataset | sigma_bins | peak_amp_dB | estimator | value_dB | deviation_dB |
|---|---|---|---|---|---|
| far | 2.000 | 20.000 | official_mean | -7.35 | 0.16 |
| far | 2.000 | 20.000 | median | -7.29 | 0.09 |
| far | 2.000 | 20.000 | mad_floor | -4.89 | 0.14 |
| far | 2.000 | 40.000 | official_mean | -7.20 | 0.32 |
| far | 2.000 | 40.000 | median | -7.27 | 0.11 |
| far | 2.000 | 40.000 | mad_floor | -4.87 | 0.17 |
| far | 6.000 | 20.000 | official_mean | -6.12 | 1.40 |
| far | 6.000 | 20.000 | median | -6.74 | 0.64 |
| far | 6.000 | 20.000 | mad_floor | -3.92 | 1.12 |
| far | 6.000 | 40.000 | official_mean | -4.72 | 2.79 |
| far | 6.000 | 40.000 | median | -6.56 | 0.82 |
| far | 6.000 | 40.000 | mad_floor | -3.51 | 1.52 |
| near | 2.000 | 20.000 | official_mean | -6.51 | 0.18 |
| near | 2.000 | 20.000 | median | -6.51 | 0.06 |
| near | 2.000 | 20.000 | mad_floor | -4.01 | 0.12 |
| near | 2.000 | 40.000 | official_mean | -6.33 | 0.36 |
| near | 2.000 | 40.000 | median | -6.50 | 0.07 |
| near | 2.000 | 40.000 | mad_floor | -3.97 | 0.16 |
| near | 6.000 | 20.000 | official_mean | -5.05 | 1.64 |
| near | 6.000 | 20.000 | median | -5.76 | 0.81 |
| near | 6.000 | 20.000 | mad_floor | -2.82 | 1.31 |
| near | 6.000 | 40.000 | official_mean | -3.41 | 3.28 |
| near | 6.000 | 40.000 | median | -5.51 | 1.06 |
| near | 6.000 | 40.000 | mad_floor | -2.31 | 1.83 |
