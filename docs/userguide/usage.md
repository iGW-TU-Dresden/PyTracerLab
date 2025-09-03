# Using ISOSIMpy
## Using the Graphical User Interface
In general, using the Graphical User Interface (GUI) is stricter and less versatile than using the package it is built on. Specifically, the app assumes a certain structure of time series data, is not scalable well to handle many different datasets, and offers limited post-processing functionality. Nevertheless, the GUI is a highly user-friendly option to performing analysis of groundwater residence time distributions using lumped parameter models.

The GUI is structured into different **Tabs**. Those **Tabs** represent the typical workflow and should be considered in their present order. The individual **Tabs** are described in more detail below.

### 1. The Input Tab
In this **Tab**, datasets are loaded and the most basic settings for subsequent modelling are made.
- select temporal resolution (yearly or monthly data in time series and model simulations)
- select tracer ($^3\mathrm{H}$ or $^14\mathrm{C}$)
- select and load tracer input time series file using the file dialog that opens up; see [here](#preparing-datasets) for details on how to prepare tracer input time series files
- select and load tracer observation time series file using the file dialog that opens up; see [here](#preparing-datasets) for details on how to prepare tracer observation time series files

**Note**: the same units of tracer concentration should be used in both the tracer input data and the observation data. Units are not checked internally. **If units are not equal, unwanted and wrong results are obtained!**

![An image of the Input Tab.](tab01.png)

### 2. The Model Tab
In this **Tab**, the different model parts are selected that are included in the simulations.
- select up to 4 model units to be used in parallel
    - available units:
        - Piston-Flow Model (**PM**)
        - Exponential Model (**EM**)
        - Exponential Piston-Flow Model (**EPM**)
        - Dispersion Model (**DM**)
    - each unit is associated with a corresponding fraction of the total system response or output; the fractions of all active units need to sum to units, otherwise an error is raised, and the model will not run
- specify if there is a steady state tracer input that should be considered for the time prior to the start of the datasets
- specify the warmup time span
    - this prepends the steady state tracer input for the time of the number of tracer half lives specified here
    - model warmup helps to remove unwanted irregularities that can appear in early phases of simulations; see [here](#model-warmup) for more details

**Note**: the steady state input value is interpreted in the same units that are used in the tracer input and observation datasets. Units are not checked internally. **If units are not equal, unwanted and wrong results are obtained!**

![An image of the Model Tab.](tab02.png)

### 3. The Parameters Tab
In this **Tab**, settings are made regarding model parameters, how they are bounded during calibration, and what current values they take.
- specify the lower bound, current value, upper bound, and calibration status for all model parameters; different model parameters are organized in rows
    - the value that is specified for a parameter will be used as its value for simple simulation and as the initial value for calibration
    - parameters that are set to *fixed* remain at their specified value during calibration

**Note**: parameter time units are always in months. Half lives are internally converted but other parameters having time units are interpreted in months.

![An image of the Parameters Tab.](tab03.png)

### 4. The Simulation Tab
In this **Tab**, simulations can be performed, model parameters can be calibrated automatically, results can be plotted, and reports can be generated.
- perform a model simulation using the current parameters
- perform model calibration
    - select a solver
    - change solver parameters (requires at lease a basic understanding of the solvers)
    - run automatic calibration
- plot results of current simulation / calibrated model simulation
- write a report including the calibrated parameters, error metrics, and other model details to a text file; uses a file dialog to store the report file

![An image of the Simulation Tab.](tab04.png)

![An example plot after parameter inference (calibration) using an MCMC sampler.](plot.png)

![An example report after parameter inference (calibration).](report.png)

(preparing-datasets)=
## Preparing Datasets
Datasets need to be prepared in a specific way in order for the app to be able to read the data. Files always have to be CSVs. The tracer input and observation time series data has to be of the same length. Time stamps which are present in the tracer input series but for which no observation is available have to be marked as missing values (see below). It is assumed that the time series do not have gaps and are processed accordingly before use in ISOSIMpy.

### Montly Data
Monthly tracer input series should have the following format:

```
# Date, Value
1996-01, 1.03
1996-02, 2.12
1996-03, 0.08
...
2009-11, 0.05
```

Instead of "# Date, Value", any other description can be used. **The first line in the file is skipped when reading!**

(model-warmup)=
## Model Warmup
