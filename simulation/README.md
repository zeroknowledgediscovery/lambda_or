# Simulation

This directory contains the simulation code used to evaluate **classical OR** versus **λ-OR** under outcome misclassification induced by score-based tail selection.

The scripts support four tasks:

1. core Monte-Carlo experiments for bias, RMSE, and coverage,
2. large parameter sweeps across \(p_{\mathrm{sel}}\), \(q_{\mathrm{sel}}\), sample size, and effect size,
3. volcano-style experiments for false discovery behavior at scale, and
4. figure generation from simulation outputs.

## Files

- `simulation_expanded.py`  
  Core simulation engine. Generates replicated naive-OR and λ-OR estimates under specified misclassification settings.

- `run_multisweep_parallel.py`  
  Parallel driver for large grid sweeps. Produces long-format and summary CSV outputs for downstream plotting.

- `volcano_simulation_nulls.py`  
  Simulates many features with small or null effects to compare discovery inflation in naive OR versus λ-OR.

- `sweep_volcano_simulation.py`  
  Runs the volcano simulation over a grid of sample sizes to show how significance scales with cohort size.

- `plotfig.py`  
  Reads simulation CSV outputs and generates manuscript-style figure panels.

## Main workflow

### 1. Core grid sweep

Run the main simulation over a grid of
\(p_{\mathrm{sel}}\), \(q_{\mathrm{sel}}\), \(n\), and \(\theta\):

```bash
python run_multisweep_parallel.py \
  --sim_module simulation_expanded.py \
  --out_csv lambdaor_outputs/multisweep_long.csv \
  --out_summary_csv lambdaor_outputs/multisweep_summary.csv \
  --procs 24 \
  --chunksize 5 \
  --write_every 50 \
  --R 16 \
  --p_grid 0.85 0.90 0.93 0.96 0.99 \
  --q_grid 0.60 0.70 0.80 0.90 0.95 \
  --n_grid 5000 10000 20000 35000 50000 \
  --theta_grid 0.25 0.4 0.7 1.1 1.4
```

This produces:

- `lambdaor_outputs/multisweep_long.csv`
- `lambdaor_outputs/multisweep_summary.csv`

### 2. Volcano experiment

Run a single large-scale discovery simulation:

```bash
python volcano_simulation_nulls.py \
  --out_dir ./ \
  --seed 3 \
  --m 20000 \
  --n_case 500000 \
  --n_ctrl 500000 \
  --beta_mean 0.06 \
  --beta_sd 0.03 \
  --eps 0.8 \
  --n_val 1000 \
  --var_extra_scale 1.0 \
  --alpha 0.01
```

### 3. Volcano sweep across sample size

```bash
python sweep_volcano_simulation.py
```

### 4. Plot manuscript figures

```bash
python plotfig.py
```

## What the simulations study

The experiments are designed to quantify:

- bias of naive OR versus λ-OR,
- RMSE under increasing misclassification,
- coverage of Wald intervals,
- false discovery inflation in large cohorts,
- sensitivity to \(q_{\mathrm{sel}}\), sample size, and target log-OR.

## Output format

Most scripts write **CSV outputs** for reproducibility and downstream plotting.  
Typical columns include:

- `method`
- `p_sel`, `q_sel`
- `n`, `theta`
- `ln_hat`, `se_hat`
- `bias`, `cover_95`
- `lam`, `detK`

## Notes

This codebase is intended specifically for the λ-OR manuscript simulations. It is built for reproducible Monte-Carlo experiments and figure generation, rather than as a general-purpose epidemiologic simulation package.
