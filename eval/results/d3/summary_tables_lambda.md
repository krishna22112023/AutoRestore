## Average scores across degradations

| Lambda | PSNR | SSIM | LPIPS | MANIQA | CLIPIQA | MUSIQ |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 63.5221 | 0.4659 | 0.3363 | 0.3630 | 0.5761 | 58.9929 |
| 0.3 | 63.8880 | 0.4659 | 0.3249 | 0.4055 | 0.6292 | 62.5810 |
| 0.5 | 63.9591 | 0.4692 | 0.3236 | 0.3993 | 0.6192 | 61.8875 |
| 0.7 | 63.9669 | 0.4827 | 0.3250 | 0.3864 | 0.5953 | 58.8094 |

### LPIPS

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.4029 | 0.4084 | 0.4095 | 0.4128 |
| haze_motion_blur_low_resolution | 0.2564 | 0.2071 | 0.1952 | 0.1943 |
| motion_blur_defocus_blur_noise | 0.4647 | 0.4645 | 0.4727 | 0.4674 |
| rain_noise_low_resolution | 0.2213 | 0.2196 | 0.2168 | 0.2256 |
| Mean | 0.3363 | 0.3249 | 0.3236 | 0.3250 |

**Ideal lambda for lpips (based on column minimum): 0.5**

### PSNR

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 63.6330 | 63.4805 | 63.3638 | 63.7555 |
| haze_motion_blur_low_resolution | 61.8815 | 63.3306 | 63.5671 | 63.3552 |
| motion_blur_defocus_blur_noise | 63.4556 | 63.4574 | 63.5202 | 63.5525 |
| rain_noise_low_resolution | 65.1182 | 65.2837 | 65.3853 | 65.2046 |
| Mean | 63.5221 | 63.8880 | 63.9591 | 63.9669 |

**Ideal lambda for psnr (based on column maximum): 0.7**

### SSIM

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.4193 | 0.4121 | 0.4048 | 0.4779 |
| haze_motion_blur_low_resolution | 0.5089 | 0.5089 | 0.5222 | 0.5154 |
| motion_blur_defocus_blur_noise | 0.4052 | 0.4072 | 0.4115 | 0.4091 |
| rain_noise_low_resolution | 0.5302 | 0.5353 | 0.5386 | 0.5285 |
| Mean | 0.4659 | 0.4659 | 0.4692 | 0.4827 |

**Ideal lambda for ssim (based on column maximum): 0.7**

### Δ QALIGN (proc − raw)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 1.7208 | 1.6986 | 1.6930 | 0.8675 |
| haze_motion_blur_low_resolution | 0.1173 | 0.8730 | 0.7737 | 0.8410 |
| motion_blur_defocus_blur_noise | 2.1597 | 2.1590 | 2.1004 | 2.1333 |
| rain_noise_low_resolution | 0.7334 | 0.7335 | 0.7163 | 0.7209 |
| Mean | 1.1828 | 1.3660 | 1.3209 | 1.1406 |

**Ideal lambda for qalign (based on column maximum): 0.3**

### Δ MANIQA (proc − raw)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.1360 | 0.1312 | 0.1355 | 0.0620 |
| haze_motion_blur_low_resolution | 0.0615 | 0.2263 | 0.2110 | 0.2244 |
| motion_blur_defocus_blur_noise | 0.2021 | 0.2064 | 0.1963 | 0.2010 |
| rain_noise_low_resolution | 0.2303 | 0.2324 | 0.2324 | 0.2362 |
| Mean | 0.1575 | 0.1991 | 0.1938 | 0.1809 |

**Ideal lambda for maniqa (based on column maximum): 0.3**

### Δ CLIPIQA (proc − raw)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.2152 | 0.2073 | 0.2118 | 0.0821 |
| haze_motion_blur_low_resolution | 0.0813 | 0.2899 | 0.2667 | 0.2893 |
| motion_blur_defocus_blur_noise | 0.0832 | 0.0918 | 0.0745 | 0.0825 |
| rain_noise_low_resolution | 0.2561 | 0.2570 | 0.2550 | 0.2586 |
| Mean | 0.1589 | 0.2115 | 0.2020 | 0.1781 |

**Ideal lambda for clipiqa (based on column maximum): 0.3**

### Δ MUSIQ (proc − raw)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 34.0193 | 33.3666 | 33.6936 | 18.8449 |
| haze_motion_blur_low_resolution | 4.5846 | 19.9469 | 18.2402 | 19.7323 |
| motion_blur_defocus_blur_noise | 41.2867 | 40.8620 | 39.5923 | 40.5304 |
| rain_noise_low_resolution | 13.9462 | 13.8702 | 13.8890 | 13.9953 |
| Mean | 23.4592 | 27.0114 | 26.3538 | 23.2757 |

**Ideal lambda for musiq (based on column maximum): 0.3**

### QALIGN (processed)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 3.7584 | 3.7374 | 3.7306 | 2.9050 |
| haze_motion_blur_low_resolution | 2.2633 | 3.0226 | 2.9197 | 2.9870 |
| motion_blur_defocus_blur_noise | 3.8442 | 3.8435 | 3.7850 | 3.8178 |
| rain_noise_low_resolution | 3.0237 | 3.0238 | 3.0066 | 3.0112 |
| Mean | 3.2224 | 3.4068 | 3.3605 | 3.1802 |

**Ideal lambda for qalign_proc (based on column maximum): 0.3**

### MANIQA (processed)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.3146 | 0.3107 | 0.3141 | 0.2407 |
| haze_motion_blur_low_resolution | 0.2906 | 0.4581 | 0.4401 | 0.4536 |
| motion_blur_defocus_blur_noise | 0.3314 | 0.3357 | 0.3256 | 0.3303 |
| rain_noise_low_resolution | 0.5153 | 0.5174 | 0.5173 | 0.5212 |
| Mean | 0.3630 | 0.4055 | 0.3993 | 0.3864 |

**Ideal lambda for maniqa_proc (based on column maximum): 0.3**

### CLIPIQA (processed)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.4424 | 0.4355 | 0.4389 | 0.3093 |
| haze_motion_blur_low_resolution | 0.5518 | 0.7615 | 0.7372 | 0.7598 |
| motion_blur_defocus_blur_noise | 0.4936 | 0.5022 | 0.4848 | 0.4928 |
| rain_noise_low_resolution | 0.8168 | 0.8178 | 0.8158 | 0.8193 |
| Mean | 0.5761 | 0.6292 | 0.6192 | 0.5953 |

**Ideal lambda for clipiqa_proc (based on column maximum): 0.3**

### MUSIQ (processed)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 61.6566 | 61.0931 | 61.3309 | 46.4823 |
| haze_motion_blur_low_resolution | 48.2000 | 63.6167 | 61.8556 | 63.3477 |
| motion_blur_defocus_blur_noise | 62.1340 | 61.7094 | 60.4397 | 61.3778 |
| rain_noise_low_resolution | 63.9808 | 63.9049 | 63.9237 | 64.0299 |
| Mean | 58.9929 | 62.5810 | 61.8875 | 58.8094 |

**Ideal lambda for musiq_proc (based on column maximum): 0.3**
