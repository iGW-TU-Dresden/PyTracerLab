# Code of the PyTracerLab paper

This folder contains the code printed in the PyTracerLab paper
(Rudolph et al., *Environmental Modelling & Software*) together with the data it uses.

| File | Content |
|---|---|
| `listing_1.py` | Listing 1 of the paper: load data, set up a model with one exponential piston-flow unit, calibrate it with differential evolution |
| `tritium_input.csv` | Tritium input time series (yearly, TU) of Example 3 of TracerLPM (Jurgens et al., 2012), as distributed with TracerLPM |
| `tritium_observations.csv` | Measured tritium concentrations (TU) of the same example; only the sampling dates are listed |

Run the listing from this folder:

```
python listing_1.py
```

`listing_1.py` is executed by the automated tests of PyTracerLab
(`tests/test_paper_listing.py`), so it always works with the current version.
The manuscript reproduces the file verbatim.

Reference: Jurgens, B.C., Böhlke, J.K., Eberts, S.M., 2012. TracerLPM (Version 1):
An Excel workbook for interpreting groundwater age distributions from environmental
tracer data. U.S. Geological Survey Techniques and Methods 4-F3.
https://pubs.usgs.gov/publication/tm4F3
