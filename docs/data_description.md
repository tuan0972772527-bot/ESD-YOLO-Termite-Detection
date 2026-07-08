# Termite Dataset Notes

This repository snapshot keeps a lightweight public sample for the five termite categories used in the study:

- `rubai`
- `huangchi`
- `heichi`
- `huangxiong`
- `heixiong`

`sample_data/` only contains qualitative examples for GitHub display. It is not the full training set.

- Each category keeps 5 non-consecutive images.
- The selected indices follow `0001`, `0030`, `0060`, `0090`, and `0120` when available in the original dataset split.
- Every sample image has its matching YOLO label file copied into `sample_data/labels/`.
- `sample_results/detection_examples/` and `sample_results/gradcam_examples/` are reserved for later result figures.

Selected sample files:

- `rubai`: `0_rubai_0001`, `0_rubai_0030`, `0_rubai_0060`, `0_rubai_0090`, `0_rubai_0120`
- `huangchi`: `1_huangchi_0001`, `1_huangchi_0030`, `1_huangchi_0060`, `1_huangchi_0090`, `1_huangchi_0120`
- `heichi`: `2_heichi_0001`, `2_heichi_0030`, `2_heichi_0060`, `2_heichi_0090`, `2_heichi_0120`
- `huangxiong`: `3_huangxiong_0001`, `3_huangxiong_0030`, `3_huangxiong_0060`, `3_huangxiong_0090`, `3_huangxiong_0120`
- `heixiong`: `4_heixiong_0001`, `4_heixiong_0030`, `4_heixiong_0060`, `4_heixiong_0090`, `4_heixiong_0120`
