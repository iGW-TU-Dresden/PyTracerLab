# import PyTracerLab
import PyTracerLab as lab

# load tracer input time series and observations (monthly time steps)
inp = lab.read_series("tritium_input_monthly.csv")
obs = lab.read_series("tritium_observations_monthly.csv", align_to=inp)

# define model (time step in years, decay constant of tritium in 1/yr)
m = lab.Model(
    dt=1 / 12,
    lambda_=lab.decay_constant("Tritium"),
    input_series=inp.values,
    target_series=obs.values,
    steady_state_input=6.0,  # tritium input before 1850 (TU)
    time_steps=inp.times,
)

# binary mixture: groundwater (exponential model) and runoff of the current
# time step (piston flow with zero travel time)
m.add_unit(lab.EMUnit(mtt=5.0), fraction=0.5, prefix="gw", bounds=[(0.5, 50.0)])
m.add_unit(lab.PMUnit(mtt=0.0), fraction=0.5, prefix="runoff")
m.set_fixed("runoff.mtt")
m.set_fixed("gw.fraction", False)  # calibrate the mixing fraction

# calibrate with differential evolution (relative measurement uncertainty 20 %)
solver = lab.Solver(m)
opt_param, opt_sim = solver.differential_evolution(sigma_rel=0.2)
print(opt_param)
