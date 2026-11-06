# import PyTracerLab
import PyTracerLab as lab

# load tracer input time series and observations
inp = lab.read_series("tritium_input.csv")
obs = lab.read_series("tritium_observations.csv", align_to=inp)

# define model (yearly time steps, decay constant of tritium in 1/yr)
m = lab.Model(
    dt=1.0,
    lambda_=lab.decay_constant("Tritium"),
    input_series=inp.values,
    target_series=obs.values,
    time_steps=inp.times,
)

# add LPM unit with initial parameter values and bounds
epm = lab.EPMUnit(mtt=10.0, eta=1.5)
m.add_unit(epm, fraction=1.0, bounds=[(0.5, 100.0), (1.0, 5.0)])

# create solver and calibrate model parameters
solver = lab.Solver(m)
opt_param, opt_sim = solver.differential_evolution()
print(opt_param)
