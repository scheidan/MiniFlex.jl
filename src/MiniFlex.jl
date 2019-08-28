## -------------------------------------------------------
##
## August 28, 2019 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------


module MiniFlex


using DifferentialEquations
using LinearAlgebra
import RecursiveArrayTools

export ModelStructure, ModelSolution
export build_model
export Q, evapotranspiration

export Param, FlowParam, EvapoParam

# -----------
# Types for model definition

# for nicer parameter definitions

const Param = RecursiveArrayTools.ArrayPartition
const FlowParam = RecursiveArrayTools.ArrayPartition
const EvapoParam = RecursiveArrayTools.ArrayPartition

@doc """
# Struct to hold model definition

## Fields
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
"""
struct ModelStructure
    routing::AbstractMatrix
    precip::Function
end



# -----------
# define systems

@doc """
# Struct to hold the ODE solution

## Fields
- `solution::DiffEqBase.AbstractODESolution`: the solution of the ODE.
                         See documentation of `DifferentialEquations.jl`.
- `θ`: model parameters used to compute solution

"""
struct ModelSolution
    solution::DiffEqBase.AbstractODESolution
    θ::AbstractArray
end

@doc """
# Set up the model and ODE solver

  build_model(mod::ModelStructure)

## Return value
A *function* to be called with arguments:

(p, V0, time, args...; kwargs...)

where
- `p`: parameters
- `V0`: vector of initial storage volumes for each reservoir
- `time`: points in time to evaluate model
- `args`: other arguments passed to `Differentialequations.solve`, e.g. `BS3()` to set solver
- `kwargs`: any keyword arguments passed to `Differentialequations.solve`, e.g. `rel.tol=1e-3`

The function returns a struct of type `ModelSolution`.

Example:

```julia
## 1) define model structure
moddef = ModelStructure(
    ## routing matrix
    [0   0  0  0;
     0.5 0  0  0;      # 0.5*Q1 -> S2
     0.5  1 0  0;      # 0.5*Q1 + 1*Q2 -> S3
     0    0  1 0],     # Q3 -> S4
    ## preciptation(t)
    precip
)

## 2) build model
my_model = build_model(moddef)

## 3) run model
p = Param(...)  # define parameters
V0 = zeros(4)   # inital storage

sol = my_model(p, V0, 0:10.0:1000) # default solver
sol2 = my_model(p, V0, 0:10.0:1000, ImplicitMidpoint(), reltol=1e-3, dt=5)
```
"""
function build_model(mod::ModelStructure)

    if(size(mod.routing, 1) != size(mod.routing, 2))
        error("Routing matrix must be square!")
    end

    # define ODE for all storages
    function dV(dV,V,p,t)

        θrun = p.x[1] # runoff parameters
        θeva = p.x[2] # evapotranspiration parameters

        # calculate flows
        outQ = Q.(V, t, θrun.x)
        dV .= mod.precip(t) .+ mod.routing*outQ
        dV .-= outQ             # substact outlow
        # substract evapotranspiration
        dV .-= evapotranspiration.(V, t, θeva.x)
    end


    function solve_model(p, V0, time, args...; kwargs...)

        # check dimensions
        n_storages = size(mod.routing, 1)

        if(!(length(p.x[1].x) == length(p.x[2].x) == n_storages))
            error("Parameter dimensions are not matching the number " *
                  "of storages ($(n_storages))!")
        end

        if(length(mod.precip(minimum(time))) != n_storages)
            error("Precipitation function must return a vector of the same length " *
                  "as the number of storages ($(n_storages))!")
        end

        if(length(V0) != n_storages)
            error("Initial values must be a vector of the same length " *
                  "as the number of storages ($(n_storages))!")
        end

        # solve ode
        prob = ODEProblem(dV,
                          eltype(p).(V0),
                          (0.0, maximum(time)),
                          p)
        sol = solve(prob, args...; saveat=time, kwargs...)
        ModelSolution(sol, p)
    end

    return solve_model
end


# -----------
# functions for flow and evapotranspiration

# N.B. THIS FUNCTION DO PROBABLY NOT MAKE MUCH SENSE!!!


function Q(V, t, θ)
    θ[1]*(1+θ[2])^V
end

@doc """
# Calculate outflow from each reservoir for give point in time

    Q(modsol::ModelSolution, time)
"""
function Q(modsol::ModelSolution, time)
    θq = modsol.θ.x[1] # runoff parameters
    hcat([Q.(modsol.solution(t), t, θq.x) for t in time]...)
end


function evapotranspiration(V, t, θ)
    θ[1]*V/(V+θ[2])
end

@doc """
# Calculate evapotranspiration for each reservoir at give point in time

    evapotranspiration(modsol::ModelSolution, time)
"""
function evapotranspiration(solution::ModelSolution, time)
    θe = modsol.θ.x[2]  # evapo parameters
    hcat([evapotranspiration.(modsol.solution(t), t, θe.x) for t in time]...)
end

end # module
