# Planck Window onto Parallel Universes

Paper and code for version 3 of Preprints.org 202104.0350 (v1 and v2 appeared as *The Gateway to Parallel Universe & Connected Physics*).

Paper: [`paper/PlanckWindowParallelUniverses.pdf`](paper/PlanckWindowParallelUniverses.pdf)

The analysis measures, on the Planck 2018 temperature maps, which bubble-collision discs and cosmic textures the sky still allows. No imprint is detected. The Cold Spot lies 13 per cent below the detection edge.

## Code

Needs Python 3 with numpy, scipy, healpy, matplotlib and Pillow. Inputs are the Planck PR3 files listed with their SHA-256 in `PLANCK_SHA256SUMS` (Planck Legacy Archive or IRSA).

- `code/coldspot/coldspot.py`: targeted fit at the Cold Spot with selection-matched simulations.
  `python3 -I coldspot.py <planck_dir> <out_dir> smica 3000 20261112`
- `code/fullsky/`: blind full-sky search (`skylib.py` shared library).
  - `prep.py <planck_dir> <data_dir> <map>` reduces a map to harmonic coefficients.
  - `run.py obs|null|inj ...` runs the observed scan, the null simulations and the injections (arguments at the top of the file).
  - `analyze.py <out_dir> <result.json>` gives thresholds, p-values, efficiencies and bounds.
- `code/window/sim_cs.py <fullsky_code> <planck_dir> <cache_dir> <amp_muK> <nsim> <seed> <out.json>`: detection of a 6 degree texture imprint at random positions.
- `code/window/figv3.py`, `code/coldspot/figures2c.py`, `code/fullsky/figures.py`: figures.

## Results

`results/` holds the numbers quoted in the paper: `coldspot/` (targeted fit), `fullsky/` (blind search, efficiencies, bounds), `window/` (imprint simulations).
