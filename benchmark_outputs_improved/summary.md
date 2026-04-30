# SOH Benchmark Summary

## Headline

- Traditional wins on predictive quality: MAE 0.631 vs 3.933, RMSE 1.986 vs 5.665, R2 0.887 vs 0.079.
- Federated wins on communication and reported memory in the current 5-satellite setup: total MB 76.5 vs 236.7, peak memory KB 117.8 vs 501.3.
- Traditional is faster end-to-end: wall time 2.35s vs 72.69s.

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

- Traditional accuracy_1pct per second: 0.1772
- Federated accuracy_1pct per second: 0.0000
- Traditional R2 per second: 0.3769
- Federated R2 per second: 0.0011

## Next Step

- Keep federated at 1 round for now, because the current random-forest aggregation method does not improve across additional rounds.
- If you want a multi-round federated curve that genuinely improves, the federated algorithm needs to evolve the global model across rounds instead of rebuilding the same ensemble.