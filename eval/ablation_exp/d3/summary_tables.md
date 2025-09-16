### LPIPS

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.4194 | 0.4195 | 0.4183 | 0.3959 |
| haze_motion_blur_low_resolution | 0.2776 | 0.2737 | 0.2685 | 0.2655 |
| motion_blur_defocus_blur_noise | 1.0161 | 1.0152 | 1.0109 | 1.0078 |
| rain_noise_low_resolution | 0.3077 | 0.3065 | 0.3103 | 0.3135 |
| Mean | 0.5052 | 0.5037 | 0.5020 | 0.4956 |

**Ideal lambda for lpips (based on column minimum): 0.7**

### PSNR

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 18.3798 | 18.2299 | 18.5376 | 19.1782 |
| haze_motion_blur_low_resolution | 15.6481 | 15.5570 | 15.6547 | 15.7871 |
| motion_blur_defocus_blur_noise | 14.9979 | 14.9348 | 15.0693 | 15.1275 |
| rain_noise_low_resolution | 16.6812 | 16.8459 | 16.8994 | 16.6754 |
| Mean | 16.4268 | 16.3919 | 16.5402 | 16.6921 |

**Ideal lambda for psnr (based on column maximum): 0.7**

### SSIM

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.5881 | 0.5858 | 0.5902 | 0.6073 |
| haze_motion_blur_low_resolution | 0.5704 | 0.5767 | 0.5735 | 0.5737 |
| motion_blur_defocus_blur_noise | 0.1444 | 0.1425 | 0.1474 | 0.1474 |
| rain_noise_low_resolution | 0.4416 | 0.4441 | 0.4439 | 0.4392 |
| Mean | 0.4361 | 0.4373 | 0.4387 | 0.4419 |

**Ideal lambda for ssim (based on column maximum): 0.7**

### Δ QALIGN (proc − raw)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 1.7208 | 1.6884 | 1.6930 | 1.6660 |
| haze_motion_blur_low_resolution | 0.8335 | 0.8730 | 0.8214 | 0.8428 |
| motion_blur_defocus_blur_noise | 2.1597 | 2.1590 | 2.1004 | 2.1333 |
| rain_noise_low_resolution | 0.7334 | 0.7335 | 0.7163 | 0.7209 |
| Mean | 1.3619 | 1.3635 | 1.3328 | 1.3407 |

**Ideal lambda for qalign (based on column maximum): 0.3**

### Δ MANIQA (proc − raw)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.1360 | 0.1307 | 0.1355 | 0.1317 |
| haze_motion_blur_low_resolution | 0.2227 | 0.2263 | 0.2187 | 0.2241 |
| motion_blur_defocus_blur_noise | 0.2021 | 0.2064 | 0.1963 | 0.2010 |
| rain_noise_low_resolution | 0.2303 | 0.2324 | 0.2324 | 0.2362 |
| Mean | 0.1978 | 0.1990 | 0.1957 | 0.1983 |

**Ideal lambda for maniqa (based on column maximum): 0.3**

### Δ CLIPIQA (proc − raw)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.2152 | 0.2073 | 0.2118 | 0.2266 |
| haze_motion_blur_low_resolution | 0.2789 | 0.2899 | 0.2778 | 0.2909 |
| motion_blur_defocus_blur_noise | 0.0832 | 0.0918 | 0.0745 | 0.0825 |
| rain_noise_low_resolution | 0.2561 | 0.2570 | 0.2550 | 0.2586 |
| Mean | 0.2083 | 0.2115 | 0.2048 | 0.2146 |

**Ideal lambda for clipiqa (based on column maximum): 0.7**

### Δ MUSIQ (proc − raw)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 34.0193 | 33.4310 | 33.6936 | 32.4519 |
| haze_motion_blur_low_resolution | 19.5647 | 19.9469 | 19.0762 | 19.8486 |
| motion_blur_defocus_blur_noise | 41.2867 | 40.8620 | 39.5923 | 40.5304 |
| rain_noise_low_resolution | 13.9462 | 13.8702 | 13.8890 | 13.9953 |
| Mean | 27.2042 | 27.0275 | 26.5628 | 26.7065 |

**Ideal lambda for musiq (based on column maximum): 0**

### QALIGN (processed)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 3.7584 | 3.7271 | 3.7306 | 3.8550 |
| haze_motion_blur_low_resolution | 2.9815 | 3.0226 | 2.9641 | 2.9860 |
| motion_blur_defocus_blur_noise | 3.8442 | 3.8435 | 3.7850 | 3.8178 |
| rain_noise_low_resolution | 3.0237 | 3.0238 | 3.0066 | 3.0112 |
| Mean | 3.4019 | 3.4043 | 3.3716 | 3.4175 |

**Ideal lambda for qalign_proc (based on column maximum): 0.7**

### MANIQA (processed)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.3146 | 0.3101 | 0.3141 | 0.3275 |
| haze_motion_blur_low_resolution | 0.4539 | 0.4581 | 0.4461 | 0.4527 |
| motion_blur_defocus_blur_noise | 0.3314 | 0.3357 | 0.3256 | 0.3303 |
| rain_noise_low_resolution | 0.5153 | 0.5174 | 0.5173 | 0.5212 |
| Mean | 0.4038 | 0.4054 | 0.4008 | 0.4079 |

**Ideal lambda for maniqa_proc (based on column maximum): 0.7**

### CLIPIQA (processed)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 0.4424 | 0.4354 | 0.4389 | 0.4678 |
| haze_motion_blur_low_resolution | 0.7486 | 0.7615 | 0.7421 | 0.7586 |
| motion_blur_defocus_blur_noise | 0.4936 | 0.5022 | 0.4848 | 0.4928 |
| rain_noise_low_resolution | 0.8168 | 0.8178 | 0.8158 | 0.8193 |
| Mean | 0.6253 | 0.6292 | 0.6204 | 0.6346 |

**Ideal lambda for clipiqa_proc (based on column maximum): 0.7**

### MUSIQ (processed)

| Degradation \ Lambda | 0 | 0.3 | 0.5 | 0.7 |
| --- | --- | --- | --- | --- |
| dark_defocus_blur_jpeg compression_artifact | 61.6566 | 61.1575 | 61.3309 | 62.7745 |
| haze_motion_blur_low_resolution | 63.2073 | 63.6167 | 62.6387 | 63.2736 |
| motion_blur_defocus_blur_noise | 62.1340 | 61.7094 | 60.4397 | 61.3778 |
| rain_noise_low_resolution | 63.9808 | 63.9049 | 63.9237 | 64.0299 |
| Mean | 62.7447 | 62.5971 | 62.0833 | 62.8639 |

**Ideal lambda for musiq_proc (based on column maximum): 0.7**
