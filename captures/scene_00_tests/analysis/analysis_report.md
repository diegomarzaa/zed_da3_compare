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

- dashboard.png
- plots/abs_error_by_object.png
- plots/da3_minus_zed_error.png
- plots/error_vs_distance.png
- plots/gt_vs_pred_scatter.png
- plots/relative_error_by_object.png
- plots/winner_counts.png

## Generated visual overlays

- visuals/sample_000003_roi_overlay.png
