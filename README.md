# MiniFlex.jl - a conceptual hydrological model with flexible routing

[![Build Status](https://travis-ci.com/scheidan/MiniFlex.jl.svg?branch=master)](https://travis-ci.com/scheidan/MiniFlex.jl)
[![Build Status](https://ci.appveyor.com/api/projects/status/github/scheidan/MiniFlex.jl?svg=true)](https://ci.appveyor.com/project/scheidan/MiniFlex-jl)
[![Coveralls](https://coveralls.io/repos/github/scheidan/MiniFlex.jl/badge.svg?branch=master)](https://coveralls.io/github/scheidan/MiniFlex.jl?branch=master)


## Installation

1. Install Julia version `>=1`.

2. Install `MiniFlex.jl` in Julia with:
```Julia
] add https://github.com/scheidan/MiniFlex.jl

```


## Usage

TODO!
The example below is mainly a placeholder.

``` julia

using MiniFlex

import Interpolations
import ForwardDiff

# -----------
# import resp. fake some data

t_rain = range(0, stop=900, length=222);
rain_obs = [ifelse(t<500, abs(sin(t/50)*15), 0.0)  for t in t_rain];
# function to interpolate rain obersvations
rain = Interpolations.LinearInterpolation(t_rain, rain_obs,
                                          extrapolation_bc = 0);
# it must return the input for each storage at time t
precip(t) = [rain(t), 0.0, 0.0, 0.0]


# -----------
# define model

# matrix defines routing from column to row
M = [0   0  0  0;
     0.5 0  0  0;      # 0.5*Q1 -> S2
     0.5 1  0  0;      # 0.5*Q1 + 1*Q2 -> S3
     0   0  1  0]      # Q3 -> S4

moddef = ModelStructure(
    M,                          # routing matrix
    precip                      # preciptation(t)
)


# -----------
# solve model

# construct function to solve model
my_model = build_model(moddef)


# define a NamedTuple for parameters
p = (θflow = [[0.1, 0.01],
              [0.05, 0.01],
               [0.02, 0.01],
               [0.01, 0.01]],
      θevap = [[0.1, 0.01],
               [0.05, 0.01],
               [0.02, 0.01],
               [0.01, 0.01]])


sol = my_model(p, zeros(4), 0:10.0:1000,
               reltol=1e-3);

# any additional arguments are passed to DifferentialEquations.solve(). E.g.
sol = my_model(p, zeros(4), 0:10.0:1000,
               ImplicitMidpoint(), reltol=1e-3, dt = 0.1)

alternatively, the model can be called with a vector. This can be
useful for optimization:
v = randn(16)
sol = my_model(v, zeros(4), 0:10.0:1000,
               reltol=1e-3);

# Note, `v` can contain values from -Inf to Inf. The parameters are
# automatically transformed to correct parameter space if needed.


# -----------
# define loss

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


# calucalte loss and the gradients
loss(v, flow_data[:,1], flow_data[:,2])
loss_grad(v, flow_data[:,1], flow_data[:,2])

```

`
