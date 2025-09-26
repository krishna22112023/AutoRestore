## Average scores across degradations

| Method | PSNR | SSIM | LPIPS | MANIQA | CLIPIQA | MUSIQ |
| --- | --- | --- | --- | --- | --- | --- |
| planned_no_optimisation | 63.8793 | 0.4719 | 0.3229 | 0.1961 | 0.2058 | 26.9772 |
| planned_opt_no_verify | 63.9316 | 0.4664 | 0.3216 | 0.1923 | 0.2088 | 27.0952 |
| random_planning | 63.9129 | 0.4739 | 0.3269 | 0.1938 | 0.2001 | 26.3808 |
| planned_opt_verify | 63.9669 | 0.4827 | 0.3250 | 0.1809 | 0.1781 | 23.2757 |
| finetuned_planned_opt_verify | 63.7702 | 0.4653 | 0.3250 | 0.1993 | 0.2091 | 26.9063 |

### LPIPS

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.4003 | 0.3994 | 0.3998 | 0.4128 | 0.4070 |
| haze_motion_blur_low_resolution | 0.2078 | 0.2159 | 0.2016 | 0.1943 | 0.2059 |
| motion_blur_defocus_blur_noise | 0.4638 | 0.4583 | 0.4734 | 0.4674 | 0.4625 |
| rain_noise_low_resolution | 0.2196 | 0.2126 | 0.2330 | 0.2256 | 0.2247 |
| Mean | 0.3229 | 0.3216 | 0.3269 | 0.3250 | 0.3250 |

**Best method for lpips (based on column minimum): planned_opt_no_verify**

### PSNR

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 63.5075 | 63.6196 | 63.5293 | 63.7555 | 63.3087 |
| haze_motion_blur_low_resolution | 63.2105 | 63.1096 | 63.5137 | 63.3552 | 63.2246 |
| motion_blur_defocus_blur_noise | 63.4557 | 63.5292 | 63.6322 | 63.5525 | 63.5194 |
| rain_noise_low_resolution | 65.3436 | 65.4682 | 64.9764 | 65.2046 | 65.0284 |
| Mean | 63.8793 | 63.9316 | 63.9129 | 63.9669 | 63.7702 |

**Best method for psnr (based on column maximum): planned_opt_verify**

### SSIM

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.4195 | 0.4182 | 0.4175 | 0.4779 | 0.4008 |
| haze_motion_blur_low_resolution | 0.5097 | 0.4915 | 0.5297 | 0.5154 | 0.5142 |
| motion_blur_defocus_blur_noise | 0.4085 | 0.4065 | 0.4123 | 0.4091 | 0.4094 |
| rain_noise_low_resolution | 0.5497 | 0.5494 | 0.5363 | 0.5285 | 0.5369 |
| Mean | 0.4719 | 0.4664 | 0.4739 | 0.4827 | 0.4653 |

**Best method for ssim (based on column maximum): planned_opt_verify**

### Δ QALIGN (proc − raw)

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 1.6963 | 1.6853 | 1.7089 | 0.8675 | 1.6737 |
| haze_motion_blur_low_resolution | 0.8512 | 0.8041 | 0.8623 | 0.8410 | 0.8549 |
| motion_blur_defocus_blur_noise | 2.1640 | 2.1175 | 2.0704 | 2.1333 | 2.1383 |
| rain_noise_low_resolution | 0.7324 | 0.6979 | 0.7280 | 0.7209 | 0.7381 |
| Mean | 1.3610 | 1.3262 | 1.3424 | 1.1406 | 1.3512 |

**Best method for qalign (based on column maximum): planned_no_optimisation**

### Δ MANIQA (proc − raw)

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.1269 | 0.1354 | 0.1296 | 0.0620 | 0.1326 |
| haze_motion_blur_low_resolution | 0.2214 | 0.2030 | 0.2151 | 0.2244 | 0.2220 |
| motion_blur_defocus_blur_noise | 0.2039 | 0.2016 | 0.1973 | 0.2010 | 0.2023 |
| rain_noise_low_resolution | 0.2323 | 0.2291 | 0.2331 | 0.2362 | 0.2403 |
| Mean | 0.1961 | 0.1923 | 0.1938 | 0.1809 | 0.1993 |

**Best method for maniqa (based on column maximum): finetuned_planned_opt_verify**

### Δ CLIPIQA (proc − raw)

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.2025 | 0.2131 | 0.2034 | 0.0821 | 0.2094 |
| haze_motion_blur_low_resolution | 0.2786 | 0.2830 | 0.2744 | 0.2893 | 0.2813 |
| motion_blur_defocus_blur_noise | 0.0897 | 0.0867 | 0.0700 | 0.0825 | 0.0858 |
| rain_noise_low_resolution | 0.2525 | 0.2523 | 0.2525 | 0.2586 | 0.2598 |
| Mean | 0.2058 | 0.2088 | 0.2001 | 0.1781 | 0.2091 |

**Best method for clipiqa (based on column maximum): finetuned_planned_opt_verify**

### Δ MUSIQ (proc − raw)

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 33.5315 | 33.3868 | 33.4151 | 18.8449 | 33.4853 |
| haze_motion_blur_low_resolution | 19.7464 | 20.8424 | 19.4782 | 19.7323 | 19.9736 |
| motion_blur_defocus_blur_noise | 40.7424 | 40.5901 | 38.9520 | 40.5304 | 39.9462 |
| rain_noise_low_resolution | 13.8885 | 13.5615 | 13.6778 | 13.9953 | 14.2201 |
| Mean | 26.9772 | 27.0952 | 26.3808 | 23.2757 | 26.9063 |

**Best method for musiq (based on column maximum): planned_opt_no_verify**

### QALIGN (processed)

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 3.7391 | 3.7155 | 3.7421 | 2.9050 | 3.7112 |
| haze_motion_blur_low_resolution | 2.9972 | 2.8784 | 3.0255 | 2.9870 | 2.9981 |
| motion_blur_defocus_blur_noise | 3.8486 | 3.8020 | 3.7478 | 3.8178 | 3.8228 |
| rain_noise_low_resolution | 3.0227 | 2.9882 | 3.0026 | 3.0112 | 3.0284 |
| Mean | 3.4019 | 3.3460 | 3.3795 | 3.1802 | 3.3901 |

**Best method for qalign_proc (based on column maximum): planned_no_optimisation**

### MANIQA (processed)

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.3072 | 0.3116 | 0.3075 | 0.2407 | 0.3113 |
| haze_motion_blur_low_resolution | 0.4506 | 0.4145 | 0.4497 | 0.4536 | 0.4505 |
| motion_blur_defocus_blur_noise | 0.3332 | 0.3309 | 0.3255 | 0.3303 | 0.3316 |
| rain_noise_low_resolution | 0.5172 | 0.5141 | 0.5228 | 0.5212 | 0.5253 |
| Mean | 0.4021 | 0.3927 | 0.4014 | 0.3864 | 0.4047 |

**Best method for maniqa_proc (based on column maximum): finetuned_planned_opt_verify**

### CLIPIQA (processed)

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.4314 | 0.4357 | 0.4278 | 0.3093 | 0.4366 |
| haze_motion_blur_low_resolution | 0.7491 | 0.7394 | 0.7493 | 0.7598 | 0.7490 |
| motion_blur_defocus_blur_noise | 0.5000 | 0.4970 | 0.4835 | 0.4928 | 0.4962 |
| rain_noise_low_resolution | 0.8132 | 0.8130 | 0.8177 | 0.8193 | 0.8205 |
| Mean | 0.6234 | 0.6213 | 0.6196 | 0.5953 | 0.6256 |

**Best method for clipiqa_proc (based on column maximum): finetuned_planned_opt_verify**

### MUSIQ (processed)

| Degradation \ Method | planned_no_optimisation | planned_opt_no_verify | random_planning | planned_opt_verify | finetuned_planned_opt_verify |
| --- | --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 61.4210 | 60.8915 | 60.8324 | 46.4823 | 61.1226 |
| haze_motion_blur_low_resolution | 63.3618 | 61.2118 | 63.5648 | 63.3477 | 63.3986 |
| motion_blur_defocus_blur_noise | 61.5898 | 61.4374 | 59.8140 | 61.3778 | 60.7935 |
| rain_noise_low_resolution | 63.9231 | 63.5961 | 63.9602 | 64.0299 | 64.2547 |
| Mean | 62.5739 | 61.7842 | 62.0428 | 58.8094 | 62.3924 |

**Best method for musiq_proc (based on column maximum): planned_no_optimisation**
