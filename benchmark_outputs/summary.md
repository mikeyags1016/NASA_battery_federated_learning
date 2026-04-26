# SOH Benchmark Summary

## Headline

- Traditional wins on predictive quality: MAE 0.852 vs 4.077, RMSE 2.956 vs 5.911, R2 0.749 vs -0.003.
- Federated wins on communication and reported memory in the current 5-satellite setup: total MB 92.0 vs 236.5, peak memory KB 162.2 vs 353.4.
- Traditional is faster end-to-end: wall time 3.57s vs 15.80s.

## Winner By Metric

- `mae`: traditional
- `rmse`: traditional
- `r2`: traditional
- `accuracy_1pct`: traditional
- `upload_mb`: federated
- `download_mb`: federated
- `total_mb`: federated
- `wall_time_s`: traditional
- `cpu_time_s`: federated
- `peak_memory_kb`: federated

## Efficiency

- Traditional accuracy_1pct per second: 0.0995
- Federated accuracy_1pct per second: 0.0016
- Traditional R2 per second: 0.2097
- Federated R2 per second: -0.0002

## Next Step

- Keep federated at 1 round for now, because the current random-forest aggregation method does not improve across additional rounds.
- If you want a multi-round federated curve that genuinely improves, the federated algorithm needs to evolve the global model across rounds instead of rebuilding the same ensemble.