## -------------------------------------------------------
## MiniFlex.jl
##
## Andreas Scheidegger -- andreas.scheidegger@eawag.ch
## -------------------------------------------------------


module MiniFlex


using DifferentialEquations
using LinearAlgebra
using TransformVariables
import StaticArrays

import Base.show

export HydroModel, ModelSolution
export build_model
export Q, evapotranspiration


# -----------
# Type for model solution

@doc """
# Struct to hold the ODE solution

## Fields
- `solution::DiffEqBase.AbstractODESolution`: the solution of the ODE.
                         See documentation of `DifferentialEquations.jl`.
- `θ`: model parameters used to compute solution

"""
struct ModelSolution
    solution::DiffEqBase.AbstractODESolution
    θ::NamedTuple
end

# pretty printing short
function Base.show(io::IO, sol::ModelSolution)
    tmin = minimum(sol.solution.t)
    tmax = maximum(sol.solution.t)
    print(io, "Model solution t ∈ [$tmin - $tmax]")
end

# pretty printin verbose
function Base.show(io::IO, ::MIME"text/plain", sol::ModelSolution)
    print(io, sol)
    println(io, "\nParameters:")
    println(io, " θflow:")
    println(io, "   $(sol.θ.θflow)")
    println(io, " θevap:")
    println(io, "   $(sol.θ.θevap)")

    println(io, "\nFlows and evapotranspiration time series can be obtained with\n"
            * " Q(solution, time_range)\n"
            * " evapotranspiration(solution, time_range)")
end

# -----------
# Type for model definition

struct HydroModel
    routing::AbstractMatrix
    precip::Function
    θtransform::TransformVariables.AbstractTransform
    dV::Function
end


function HydroModel(routing, precip)

    if(size(routing, 1) != size(routing, 2))
        error("Routing matrix must be square!")
    end

    # define parameter transformation: Real vector -> NamedTuple
    N = size(routing, 1)
    θtransform = as((
        θflow = as(Vector, as(Vector, asℝ₊, 2), N),
        θevap = as(Vector, as(Vector, asℝ₊, 2), N)
    ))

    # convert routing matrix to static Array
    srouting = StaticArrays.SMatrix{N,N}(routing)

    # define ODE for all storages (p is a NamedTuple)
    function dV(dV,V,p,t)
        # calculate flows
        outQ = Q.(V, t, p.θflow)
        dV .= precip(t) .+ routing*outQ
        dV .-= outQ             # substact outlow
        # substract evapotranspiration
        dV .-= evapotranspiration.(V, t, p.θevap)
    end

    HydroModel(srouting, precip, θtransform, dV)
end


@doc """
# Define and run a hydrological model

## Model definition

  HydroModel(routing::AbstractMatrix, precip::Function)

where
- `routing::AbstractMatrix`: adjacency matrix to define connection between reservoirs
- `precip::Function`: function that returns a vector of rain intensities
                      for  each reservoir at any given time point. Takes time as argument.


### Example of routing matrix

We have four reservoirs S1, S2, S3, S4:
```julia
[0    0  0  0;      # S1 has no inflow from other reservoirs
 0.5  0  0  0;      # 0.5*Q1 -> S2
 0.5  1  0  0;      # 0.5*Q1 + 1*Q2 -> S3
 0    0  1  0]      # Q3 -> S4
```

## Run a HydroModel

Objects of type `HydroModel` are callable to tun the simulation:

  (m::HydroModel)(p::NamedTuple, V0, time, args...; kwargs...)

where
- `p`: parameters. Either a NamedTuple or a Vector ∈ ℝᴺ
- `V0`: vector of initial storage volumes for each reservoir
- `time`: points in time to evaluate model
- `args`: other arguments passed to `Differentialequations.solve`, e.g. `BS3()` to set solver
- `kwargs`: any keyword arguments passed to `Differentialequations.solve`, e.g. `rel.tol=1e-3`

The return value is a struct of type `ModelSolution`.

Example:

```julia
## 1) define model

my_model = HydroModel(
    ## routing matrix
    [0   0  0  0;
     0.5 0  0  0;      # 0.5*Q1 -> S2
     0.5  1 0  0;      # 0.5*Q1 + 1*Q2 -> S3
     0    0  1 0],     # Q3 -> S4
    ## preciptation(t)
    precip
)

## 2) run model

p = (θflow = [[0.1, 0.01],    # Q1
              [0.05, 0.01],   # Q2
               [0.02, 0.01],  # Q3
               [0.01, 0.01]], # Q4
      θevap = [[0.1, 0.01],   # evapo 1
               [0.05, 0.01],  # evapo 2
               [0.02, 0.01],  # evapo 3
               [0.01, 0.01]]) # evapo 4

V0 = zeros(4)   # inital storage

sol = my_model(p, V0, 0:10.0:1000) # default solver
sol2 = my_model(randn(16), V0, 0:10.0:1000) # call with parameter vector
sol3 = my_model(p, V0, 0:10.0:1000, ImplicitMidpoint(), reltol=1e-3, dt=5)

plot(sol) # requires `Plots` to by loaded

# extract run-off and evapotranspiration
t_obs = 10:2.5:100
Q(sol, t_obs)
evapotranspiration(sol, t_obs)
```
"""
function (m::HydroModel)(p::NamedTuple, V0, time, args...; kwargs...)

    # check dimensions
    n_storages = size(m.routing, 1)

    if(!(length(p.θflow) == length(p.θevap) == n_storages))
            error("Parameter dimensions are not matching the number " *
                  "of storages ($(n_storages))!")
    end

    if(length(m.precip(minimum(time))) != n_storages)
        error("Precipitation function must return a vector of the same length " *
              "as the number of storages ($(n_storages))!")
    end

    if(length(V0) != n_storages)
        error("Initial values must be a vector of the same length " *
              "as the number of storages ($(n_storages))!")
    end

    # solve ode
    prob = ODEProblem(m.dV,
                      nested_eltype(p).(V0),
                      (minimum(time), maximum(time)),
                      p)
    sol = solve(prob, args...; saveat=time, kwargs...)
    ModelSolution(sol, p)
end

function (m::HydroModel)(p::AbstractArray, V0, time, args...; kwargs...)
    # if called with vector, we transform the parameters to tuple
    m(m.θtransform(p), V0, time, args...; kwargs...)

end


# pretty printing short
function Base.show(io::IO, m::HydroModel)
    print(io, "HydroModel ($(size(m.routing,1)) reservoirs)")
end

# pretty printin verbose
function Base.show(io::IO, ::MIME"text/plain", m::HydroModel)

    mstr = "$(m.routing)"
    mstr = replace(mstr, "0.0" => " ⋅ ")
    mstr = replace(mstr, ";" => "\n  ")
    mstr = replace(mstr, "[" => "   ")
    mstr = replace(mstr, "]" => "")

    println(io, "HydroModel model with $(size(m.routing,1)) reservoirs")
    println(io, "and routing matrix:")
    print(io, mstr)
end


# -----------
# functions for flow and evapotranspiration

# N.B. THIS FUNCTION DO PROBABLY NOT MAKE MUCH SENSE!!!


function Q(V, t, θ)
    θ[1]*(1+θ[2])^V
end

@doc """
# Calculate outflow from each reservoir for give point in time

    Q(sol::ModelSolution, time)
"""
function Q(modsol::ModelSolution, time)
    hcat([Q.(modsol.solution(t), t, modsol.θ.θflow) for t in time]...)
end


function evapotranspiration(V, t, θ)
    θ[1]*V/(V+θ[2])
end

@doc """
# Calculate evapotranspiration for each reservoir at give point in time

    evapotranspiration(sol::ModelSolution, time)
"""
function evapotranspiration(modsol::ModelSolution, time)
    hcat([evapotranspiration.(modsol.solution(t), t, modsol.θ.θevap) for t in time]...)
end



# -----------
# helpers

function nested_eltype(x)
    y = eltype(x)
    while y <: AbstractArray
        y = eltype(y)
    end
    return(y)
end

end # module
