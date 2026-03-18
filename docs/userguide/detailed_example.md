# Using the GUI: a Detailed Example

This example demonstrates the GUI functionality. We use pre-existing data of tracer input and observations that were synthetically generated (see (Example 5)[../examples/example_05.ipynb]) but the exact same steps can be carried out with any data you may have on hand. This example here considers the case of two tracers (tritium and Kr-85). Observations of tracer concentrations in groundwater are available for a number of dates, where always both tracer concentrations were measured. The GUI can also handle cases where, at certain dates, only one of the tracers is observed. See the [user guide](usage.md) for more information

```{tip}
Refer to this guide when encountering problems with the GUI. Most of the functionality of the GUI is covered here, which should answer most of the questions that may come up.
```

## Loading Data
We use tritium and Kr-85 input (examples/example_input_series_2tracer.csv) and output (examples/example_observation_series_2tracer.csv) data and perform the basic model settings in the input tab:
1. Set the model resolution to `Monthly`. This settings decides on how the input data should look like and vice versa.
2. Select `Tritium` as the first tracer; corresponding data is in the **first column** of model input and observation data.
3. Select `Krypton-85` as the second tracer; corresponding data is in the **second column** of model input and observation data.
4. Select the input file `example_input_series_2tracer.csv` using the file dialog that opens when you click on the button.
5. Select the observation file `example_observation_series_2tracer.csv` using the file dialog that opens when you click on the button.

```{tip}
It is also possible to manually enter observation data in the GUI. If you want to use this feature, **do not** enter an obervation file before. This should then replace step 5 from before.
```

![An image of the Input Tab.](tab02.png)
