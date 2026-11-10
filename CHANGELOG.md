# Changelog

All notable changes to PyTracerLab are listed here. Versions follow the tags and
[releases](https://github.com/iGW-TU-Dresden/PyTracerLab/releases) on GitHub and the
packages on [PyPI](https://pypi.org/project/PyTracerLab/).

## 0.3.0 (unreleased)

The version described in the PyTracerLab paper (Rudolph et al., *Environmental Modelling &
Software*). Some results differ from version 0.2.9; see **Changed results** below.

### Added

- **Package-level API:** `import PyTracerLab as lab` gives direct access to `Model`, `Solver`,
  all units, `read_series`, `decay_constant` and `TRACERS`. `import PyTracerLab.model` keeps
  working.
- **Reading data files:** `lab.read_series(path, align_to=None)` reads tracer input time series
  and observations from CSV or other text files (comma, semicolon, tab or space delimiters,
  decimal comma, common date formats, comment lines). Observations can be listed at their
  sampling dates only (`align_to=inp` assigns each sample to the time step of the input series).
  Clear error messages name the line and value that cannot be read. The GUI uses the same
  reader. Templates are in `docs/examples/templates`.
- **Tracer properties** (e.g. half-lives) in the package: `lab.decay_constant("Tritium")`,
  `lab.TRACERS`.
- **Measurement uncertainty per observation:** all solvers accept `sigma` (one value, one value
  per tracer or one value per observation, e.g. from `<tracer>_sd` columns of the observation
  file) and `sigma_rel` (relative uncertainty). In the GUI, one *Measurement uncertainty*
  setting applies to all solvers; plots show error bars.
- **Calibration of the mixing fraction** of binary mixtures (two units): parameter
  `<prefix of the first unit>.fraction`, fixed by default. PyTracerLab warns if there are more
  free parameters than observations; the GUI shows a note if the fraction and both mean travel
  times are free.
- **New units:** gamma model (`GMUnit`), linear model (`LMUnit`) and partial exponential model
  (`PEMUnit`), also in the GUI (GM, LM, PEM).
- **Age-bin model** (`AgeBinUnit`, Python only): a travel time distribution without a
  prescribed shape (histogram with fixed bin edges and calibrated weights).
- **DREAM outlier check:** during burn-in, chains that are stuck in a region of low posterior
  density are moved to the best chain (Vrugt et al., 2008); on by default
  (`outlier_check=True`), number of moved chains in `n_outlier_resets`.
- **Paper examples:** scripts, data and step-by-step notebooks of the five examples of the
  paper in `docs/examples/paper`.

### Changed results

- **Exponential-piston flow model (EPM):** each time step now receives the exact integral of the
  travel time distribution over the time step, as in TracerLPM. With yearly time steps and the
  steep tritium bomb peak, the previous version deviated from TracerLPM by up to 79 TU; now by
  less than 0.3 TU. `eta = 1` gave a zero response and now gives the exponential model.
- **Water older than the input series:** its contribution is calculated exactly from the
  steady-state input and the travel time distribution, instead of from a simulated warm-up
  period. There is no limit on travel times or half-lives any more (the warm-up was limited to
  5000 time steps). The arguments `n_warmup_half_lives` and `n_warmup_steps` are still accepted
  but have no effect; the GUI has no warm-up field any more.
- **No steady-state input given:** water that recharged before the start of the input series is
  assumed to contain no tracer, as before, but PyTracerLab now shows a note (GUI) or warning
  (Python). Pass `steady_state_input=0` to confirm this.
- **Piston flow model with `mtt = 0`:** the water now carries the input of the current time
  step (previously the unit returned zero); a negative `mtt` is an error.
- **MCMC and DREAM without measurement uncertainty:** the unknown standard deviation is now
  integrated out separately for each tracer (previously one common value for all tracers, so
  that the tracer with the largest values dominated).
- **Default number of DREAM chains:** `max(4, 2 d, 2 n_diff_pairs + 1)` for `d` free parameters
  (previously `max(3, 2 d)`, which failed with the default `n_diff_pairs=3` for one to three free
  parameters); the GUI uses 8 chains by default (previously 3). With fewer than 4 chains, the
  outlier check cannot work, and PyTracerLab warns.

### Fixed

- **Piston flow model with time steps other than 1:** the output was scaled by 1/`dt`. This
  affected the **GUI with yearly data**, which works in months (`dt` = 12): the piston flow
  model gave 1/12 of the correct concentration.
- **GUI, DREAM:** the default settings (3 chains, 2 difference pairs) stopped with the error
  "n_diff_pairs too large for the number of chains".
- **GUI, reading files:** files with semicolons were loaded empty, and rows with unreadable
  dates were skipped without a message.
- **Observations not on the time steps of the input series** gave an infinite objective without
  an error message; there is now a clear error (or use `read_series(..., align_to=...)`).
- **Solvers and measurement uncertainty:** differential evolution silently ignored an
  uncertainty array of the wrong size; least squares with one uncertainty per tracer assumed
  that all tracers were observed at the same dates; integer uncertainties were ignored; MCMC
  and DREAM in the GUI silently dropped uncertainties containing missing values.
- **GUI:** *Reset to defaults* in the solver dialog crashed; edits in the Parameters tab were
  lost when the Model or Input tab changed.
- **Reports:** `Model.write_report` failed without observations, and a single tracer name
  given as a string was split into characters.
- **Stable tracers** (decay constant 0) caused an `OverflowError` when the length of the warm-up
  period was calculated.

### Requirements

- Python 3.10 or newer, SciPy 1.15 or newer.

### Documentation

- User guide (English and German): data file format and templates, measurement uncertainty,
  model units, mixing fractions, water older than the input series, choosing a solver and
  checking MCMC convergence.
- Examples: Tutorials 1–6 updated and re-run; the former examples 7–9 are replaced by the paper
  examples (their pages no longer exist).
