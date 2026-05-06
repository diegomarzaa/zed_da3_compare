# scene_00_tests analysis

Annotations rows: 9
Existing sample folders: 1
Rows with existing samples: 6
Missing/stale samples referenced by scene_annotations.csv: sample_000001

## Summary by method

| method   | n | mean_pred_m | mean_abs_error_m | median_abs_error_m | max_abs_error_m | mean_rel_error | median_rel_error | mean_valid_ratio |
| -------- | - | ----------- | ---------------- | ------------------ | --------------- | -------------- | ---------------- | ---------------- |
| da3_mono | 6 | 1.4418      | 0.3668           | 0.2325             | 0.8093          | 0.3722         | 0.3751           | 1.0000           |
| zed      | 6 | 1.0782      | 0.0968           | 0.0639             | 0.3000          | 0.1054         | 0.0565           | 0.7452           |

## Winner counts

| method   | wins |
| -------- | ---- |
| zed      | 5    |
| da3_mono | 1    |

## ZED reference summary

| sample_id     | method   | joint_valid_ratio | zed_valid_ratio | method_valid_ratio | raw_mae | raw_rmse | raw_abs_rel | raw_bias | median_scale_to_zed | scaled_mae | pair_corr | pair_grad_mae |
| ------------- | -------- | ----------------- | --------------- | ------------------ | ------- | -------- | ----------- | -------- | ------------------- | ---------- | --------- | ------------- |
| sample_000003 | da3_mono | 0.8761            | 0.8761          | 1.0000             | 0.5953  | 0.7406   | 0.3641      | 0.3610   | 0.6705              | 0.3769     | 0.8406    | nan           |

## ZED reference ROI metrics

| sample_id     | object_name          | method   | zed_median_m | method_median_m | diff_m  | abs_diff_m | rel_abs_diff_to_zed | ratio_method_over_zed | zed_valid_ratio | method_valid_ratio | notes               |
| ------------- | -------------------- | -------- | ------------ | --------------- | ------- | ---------- | ------------------- | --------------------- | --------------- | ------------------ | ------------------- |
| sample_000003 | cbotella             | da3_mono | 0.2321       | 0.3566          | 0.1245  | 0.1245     | 0.5363              | 1.5363                | 0.7189          | 1.0000             | nan                 |
| sample_000003 | boton_amarillo_mando | da3_mono | 0.4400       | 0.6720          | 0.2320  | 0.2320     | 0.5272              | 1.5272                | 0.6910          | 1.0000             | muy aproximadamente |
| sample_000003 | monitor              | da3_mono | 1.0800       | 0.9287          | -0.1513 | 0.1513     | 0.1401              | 0.8599                | 0.0610          | 1.0000             | nan                 |
| sample_000003 | sillanegra           | da3_mono | 0.9752       | 1.3029          | 0.3277  | 0.3277     | 0.3360              | 1.3360                | 1.0000          | 1.0000             | muy aprox           |
| sample_000003 | caja                 | da3_mono | 1.2846       | 2.2293          | 0.9447  | 0.9447     | 0.7354              | 1.7354                | 1.0000          | 1.0000             | nan                 |
| sample_000003 | puertapared          | da3_mono | 2.4571       | 3.1615          | 0.7044  | 0.7044     | 0.2867              | 1.2867                | 1.0000          | 1.0000             | nan                 |

## Per-object recomputed metrics

| sample_id     | object_name          | gt_distance_m | zed_median_m | zed_abs_error_m | da3_mono_median_m | da3_mono_abs_error_m | winner_recomputed | notes               |
| ------------- | -------------------- | ------------- | ------------ | --------------- | ----------------- | -------------------- | ----------------- | ------------------- |
| sample_000003 | cbotella             | 0.2400        | 0.2321       | 0.0079          | 0.3566            | 0.1166               | zed               | nan                 |
| sample_000003 | boton_amarillo_mando | 0.4500        | 0.4400       | 0.0100          | 0.6720            | 0.2220               | zed               | muy aproximadamente |
| sample_000003 | monitor              | 0.7800        | 1.0800       | 0.3000          | 0.9287            | 0.1487               | da3_mono          | nan                 |
| sample_000003 | sillanegra           | 1.0600        | 0.9752       | 0.0848          | 1.3029            | 0.2429               | zed               | muy aprox           |
| sample_000003 | caja                 | 1.4200        | 1.2846       | 0.1354          | 2.2293            | 0.8093               | zed               | nan                 |
| sample_000003 | puertapared          | 2.5000        | 2.4571       | 0.0429          | 3.1615            | 0.6615               | zed               | nan                 |

## Generated plots

- plots/abs_error_by_object.png
- plots/da3_minus_zed_error.png
- plots/error_vs_distance.png
- plots/gt_vs_pred_scatter.png
- plots/relative_error_by_object.png
- plots/winner_counts.png
- plots/zed_ref_roi_abs_diff_boxplot.png
- plots/zed_ref_roi_scatter.png
- plots/zed_ref_roi_signed_diff.png
- plots/zed_ref_sample_mae_vs_validity.png
- plots/zed_ref_sample_scale.png

## Generated visual overlays

- visuals/sample_000003_roi_overlay.png