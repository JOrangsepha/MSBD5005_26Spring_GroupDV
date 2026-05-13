# Task 3 — Composite-Score Time Evolution

Code and result figure for the rising-star composite-score trajectory analysis on the Oceanus Folk subgraph of MC1.

## Contents

| File | Description |
|------|-------------|
| `time_dim_charts.py` | Recomputes the 9 rising-star inputs and the multiplicative composite at every career age, then renders the static figure and an interactive D3 page. |
| `report_fig_t_trajectory.png` | Result figure: composite score vs. career age for 7 tracked entities. |
| `trajectory_tails.html` | D3.js interactive version of the same data. |
| `data/artists.json` | Preprocessed artist records consumed by `time_dim_charts.py`. |

## Run

```bash
python time_dim_charts.py
```

Outputs `report_fig_t_trajectory.png` and `trajectory_tails.html` in the working directory.

## Tracked entities

Sailor Shift, Embers of Wrath, Orla Seabloom (baselines); Ivy Echos (Sailor Shift's earlier band, for narrative context); Copper Canyon Ghosts, Daniel O'Connell, Beatrice Albright (predicted rising stars). Sailor Shift's terminal composite of 0.4405 matches the project's original preprocessing pipeline as a reproducibility check.
