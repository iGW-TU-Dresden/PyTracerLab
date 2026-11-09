# Code of the PyTracerLab paper

This folder contains the code of the examples in the PyTracerLab paper
(Rudolph et al., *Environmental Modelling & Software*) together with the data
they use. All results in the paper were produced with PyTracerLab v0.3.0.

Run a script from this folder, e.g.

```
python example_1.py
```

Figures and tables are written to `output/`. The scripts with Markov chain Monte
Carlo runs accept `--quick` (a few iterations only, for testing). All scripts
are executed by the automated tests of PyTracerLab (`tests/test_paper_listing.py`,
`tests/test_paper_examples.py`), so they always work with the current version.

| Script | Content |
|---|---|
| `listing_1.py` | Listing 1 of the paper (reproduced verbatim): load data, set up a binary mixture with a calibrated mixing fraction, calibrate it with differential evolution (data of Example 4) |
| `example_1.py` | Example 1: forward simulations with four models and three parameter sets each, compared with TracerLPM |
| `example_2.py` | Example 2: recovery of known parameters from synthetic observations with all four solvers; run times |
| `example_3.py` | Example 3: river water from a groundwater flow model (MODFLOW 6, synthetic input); exponential model with calibrated mixing fraction and age-bin model, compared with the travel times from particle tracking |
| `example_4.py` | Example 4: measured tritium time series of the Missouri River (Example 3 of TracerLPM), compared with TracerLPM; parameter uncertainty with DREAM |

| Data file | Content |
|---|---|
| `tritium_input.csv` | Tritium input time series (yearly, TU, 1850-2020) of Example 3 of TracerLPM (Jurgens et al., 2012), as distributed with TracerLPM |
| `tritium_input_monthly.csv` | The same yearly values, used for every month of the year (the step function that TracerLPM uses for yearly input) |
| `tritium_input_semiannual.csv` | Tritium input time series of the same example with semi-annual resolution (1900-2020) |
| `tritium_observations.csv` | Measured tritium concentrations (TU) in the Missouri River of the same example (35 samples, 1963-1997); only the sampling years are listed |
| `tritium_observations_monthly.csv` | The same observations with their sampling dates (1 July of each year) |
| `example_1_tracerlpm_set1.csv` ... `set3.csv` | TracerLPM simulations (PM, EM, EPM, DM) for the three parameter sets of Example 1 |
| `example_3_input.csv` | Synthetic tracer input in recharge of the MODFLOW model of Example 3 (monthly, 1970-2019; no tracer before 1970) |
| `example_3_observations.csv` | Tracer concentrations in river water simulated with the MODFLOW model (every 5 years; all other months empty) |
| `example_3_particle_travel_times.csv` | Travel times (days) and flux weights of the particles from particle tracking (groundwater only; particles reaching the infiltration well are excluded) |
| `example_4_tracerlpm.csv` | TracerLPM simulation at its best fit for Example 4 (BMM-EMM-PFM with a PFM of age 0: mean travel time 6.8 yr, fraction 0.56), sampling dates only |

Reference: Jurgens, B.C., Böhlke, J.K., Eberts, S.M., 2012. TracerLPM (Version 1):
An Excel workbook for interpreting groundwater age distributions from environmental
tracer data. U.S. Geological Survey Techniques and Methods 4-F3.
https://pubs.usgs.gov/publication/tm4F3
