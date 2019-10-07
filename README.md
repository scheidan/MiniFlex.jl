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

TODO
The example below is mainly a placeholder.

``` julia

using MiniFlex

import Interpolations
import ForwardDiff
import StaticArrays

using Plots

# -----------
# import resp. fake some data

t_rain = range(0, stop=900, length=222);
rain_obs = [ifelse(t<500, abs(sin(t/50)*15), 0.0)  for t in t_rain];
# function to interpolate rain obersvations
rain = Interpolations.LinearInterpolation(t_rain, rain_obs,
                                          extrapolation_bc = 0);
# it must return the input for each storage at time t
precip(t) = StaticArrays.@SVector [rain(t), 0.0, 0.0, 0.0]  # using SVector avoids allocations


# -----------
# define model

my_model = HydroModel(
    [Connection(:S1 => [:S2, :S3]),  # outflow from :S1 to :S2 and :S3
     Connection(:S2 => :S3),
     Connection(:S3 => :S4)],

    P_rate = precip,
    PET_rate = x -> zeros(4)  # no evapotranspiration
)


my_model

# -----------
# solve model


# define a NamedTuple for parameters
p = (θflow = ([0.1, 0.01],
              [0.5, 0.01],
              [0.2, 0.01],
              [0.01, 0.01]),
     θevap = ([1.0, 20.0],
              [2.2, 20.0],
              [3.3, 20.0],
              [2.2, 20.0]),
     θrouting = ([0.3, 0.7], # connection from :S1
                 [1.0],      # connection from :S2
                 [1.0])      # connection from :S3
     )


sol = my_model(p, zeros(4), 0:10.0:1000,
               reltol=1e-3);

sol

# extract runoff for each reservoir
t_obs = 0:22.:1000
Q(sol, t_obs)


# or plot it
plot(sol, value="Q")
plot(sol, value="volume")


# any additional arguments are passed to DifferentialEquations.solve(). E.g.
sol = my_model(p, zeros(4), 0:10.0:1000,
               ImplicitMidpoint(), reltol=1e-3, dt = 0.1)

# Alternatively, the model can be called with a vector. This is
# useful for optimization:
v = randn(16)
sol = my_model(v, zeros(4), 0:10.0:1000,
               reltol=1e-3);

# Note, `v` can contain values from -Inf to Inf. The parameters are
# automatically transformed to the correct parameter space.


# -----------
# Automatic Differentation with ForwardDiff

# loss function
function loss(v, t_obs, Q_obs)

    # initial bucket volume
    V_init = zeros(4)

    # solve ODE
    sol = my_model(v, V_init, t_obs)

    # get Q3
    Q3 = Q(sol, t_obs)[3,:]

    # compute loss
    sum((Q3 .- Q_obs).^2)
end

# derive gradient function
loss_grad(p, t_obs, Q_obs) = ForwardDiff.gradient(p -> loss(p, t_obs, Q_obs), p)


# calculate loss and the gradients
loss(v, flow_data[:,1], flow_data[:,2])
loss_grad(v, flow_data[:,1], flow_data[:,2])

```

`
