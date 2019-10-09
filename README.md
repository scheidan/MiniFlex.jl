# MiniFlex - A Conceptual Hydrological Model for Automatic Differentation

[![Build Status](https://travis-ci.com/scheidan/MiniFlex.jl.svg?branch=master)](https://travis-ci.com/scheidan/MiniFlex.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/scheidan/MiniFlex.jl?svg=true)](https://ci.appveyor.com/project/scheidan/MiniFlex-jl)
[![Coveralls](https://coveralls.io/repos/github/scheidan/MiniFlex.jl/badge.svg?branch=master)](https://coveralls.io/github/scheidan/MiniFlex.jl?branch=master)


## Motivation

`MiniFlex` is a conceptual hydrological model formulated continuously
in time based on differential equations. The most important design goals are:

- Allow for any number of reservoirs and arbitrary routing.
- Make use of efficient ODE solvers provided by
  `DifferentialEquations.jl`.
- Ensure compatibility with Automatic Differentation libraries such as
  `ForwardDiff.jl` to experiment with more efficient optimization algorithms
    and Hamiltonian MC.
- Aim for fast run time.


## Installation

1. Install Julia version `>=1`.

2. Install `MiniFlex.jl` in Julia with:
```Julia
] add https://github.com/scheidan/MiniFlex.jl

```


## Usage

### Preparation of Input Data

As `MiniFlex` is defines as continuous model without predefined
time-steps, the precipitation and potential evapotranspiration _rates_
must be available for every point in time. This means observed
time-series must be interpolated.

`MiniFlex` expects a function that if evaluated at time `t` returns a
vector with the input rates for every reservoir.

``` julia
using Interpolations
using StaticArrays: @SVector


# simulate some observations
t_precip = range(0, stop=900, length=3*365);
obs_precip = [t<500 ? abs(sin(t/50)*15) : 0.0  for t in t_precip]

# returns a function that interpolates the rain observations
rain = LinearInterpolation(t_precip, obs_precip, extrapolation_bc = 0)

# Our model will have 4 reservoirs but only
# the first one obtains precipitations
P_rate(t) = @SVector [rain(t), 0.0, 0.0, 0.0]

# For the similicity of the example we assume
# no potential evapotranspiration
PET_rate(t) = @SVector zeros(4)
```

### Model definition

Next we define the model i.e. the reservoirs and there connections:

``` julia
using MiniFlex

my_model = HydroModel(
    connections = [Connection(:S1 => [:S2, :S3])
	               Connection(:S2 => :S3),
                   Connection(:S3 => :S4)],
    P_rate = P_rate,
    PET_rate = PET_rate
)

```
Of course different reservoir names than `S1`, `S2`, ... can be used.

If you print the `my_model`  the order of the reservoirs is shown
(always alphabetically). Make sure this matches `P_rate` and `PET_rate`.

### Solve model

First we define the model parameters:
``` julia

# parameters must be a NamedTuple with the following structure
p = (θflow = ([0.1, 0.01],
              [0.5, 0.01],
              [0.2, 0.01],
              [0.01, 0.01]),
     θevap = ([1.0, 20.0],
              [2.2, 20.0],
              [3.3, 20.0],
              [2.2, 20.0]),
     θrouting = ([0.3, 0.7], # from S1 -> 30% to S2, 70% to S3
                 [1.0],      # from S2 -> 100% to S3
                 [1.0])      # from S3 -> 100% to S4
     )

```

Then solve the model an extract the outflows:

```julia
V_init = zeros(4)
sol = my_model(p, V_init, 0:1000)

# extract runoff for each reservoir at t_obs.
# Note, any points in time are possible
t_obs = 0:22.3:1000
Q(sol, t_obs)

```

The model outputs can be plotted:
```julia

using Plots

plot(sol, value="Q")
plot(sol, value="Q", xlims=(100,200), legend=false)
plot(sol, value="volume")

```

You can pass additional arguments to influence the ODE solver,
see [here](http://docs.juliadiffeq.org/latest/basics/common_solver_opts.html). For example:

```julia
my_model(p, V_init, 0:1000, saveat=0:50:1000
               ImplicitMidpoint(), reltol=1e-3, dt = 0.1)

```

Optimization and sampling routines usually expect a function that can
be called by a vector.  Therefore a model can also be called by a
parameter vector.

Note, any vector in ℝⁿ is allowed, as parameters are automatically transformed to the
correct parameter space. For example it is ensured that the outflow
fractions always add up to one.

```julia
v = randn(17)
sol = my_model(v, V_init, 0:1000);


sol.θ    # parameter tuple corresponding to `v`
```


### Automatic Differentation
Using Automatic Differentation libraries (currently only tested with [`ForwardDiff`](http://www.juliadiff.org/ForwardDiff.jl/stable/)), it is
possible to compute the gradient of the solution with respect to the
parameters:
```julia
using ForwardDiff: gradient

# simulate some "observations" for outflow of S3
flow_data = collect(hcat([ifelse(t < 600, [t,50.0], [t, 0.0]) for t in 1:10:1000]...)')


# define loss function
function loss(p, t_obs, Q_obs)

    # solve ODE
    V_init = zeros(4)
    sol = my_model(p, V_init, t_obs, saveat=t_obs)

    # get Q3
    Q3 = Q(sol, t_obs)[3,:]

    # compute loss
    sum((Q3 .- Q_obs).^2)
end

# derive gradient function
loss_grad(v, t_obs, Q_obs) = gradient(v -> loss(p, t_obs, Q_obs), p)

# calculate loss
loss(v, flow_data[:,1], flow_data[:,2])
# and the respective gradient
loss_grad(v, flow_data[:,1], flow_data[:,2])
```
