## -------------------------------------------------------
##
## August 28, 2019 -- Andreas Scheidegger
## andreas.scheidegger@eawag.ch
## -------------------------------------------------------


module MiniFlex


using DifferentialEquations
using LinearAlgebra
using TransformVariables

export ModelStructure, ModelSolution
export build_model
export Q, evapotranspiration

export Param, FlowParam, EvapoParam

# -----------
# Types for model definition

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
    θtransform::TransformVariables.AbstractTransform
end

function ModelStructure(routing, precip)

    if(size(routing, 1) != size(routing, 2))
        error("Routing matrix must be square!")
    end

    # define parameter transformation: Real vector -> NamedTuple
    N = size(routing, 1)
    θtransform = as((
        θflow = as(Vector, as(Vector, asℝ₊, 2), N),
        θevap = as(Vector, as(Vector, asℝ₊, 2), N)
    ))

    ModelStructure(routing, precip, θtransform)
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
    θ::NamedTuple
end


@doc """
# Set up the model and ODE solver

  build_model(mod::ModelStructure)

## Return value
A *function* to be called with arguments:

(p, V0, time, args...; kwargs...)

where
- `p`: parameters. Either a NamedTuple or a Vector ∈ ℝ
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
 # define parameters
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
sol2 = my_model(p, V0, 0:10.0:1000, ImplicitMidpoint(), reltol=1e-3, dt=5)
```
"""
function build_model(mod::ModelStructure)

    # define ODE for all storages (p is a NamedTuple)
    function dV(dV,V,p,t)
        # calculate flows
        outQ = Q.(V, t, p.θflow)
        dV .= mod.precip(t) .+ mod.routing*outQ
        dV .-= outQ             # substact outlow
        # substract evapotranspiration
        dV .-= evapotranspiration.(V, t, p.θevap)
    end

    function solve_model(p::NamedTuple, V0, time, args...; kwargs...)

        # check dimensions
        n_storages = size(mod.routing, 1)

        if(!(length(p.θflow) == length(p.θevap) == n_storages))
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
                          nested_eltype(p).(V0),
                          (0.0, maximum(time)),
                          p)
        sol = solve(prob, args...; saveat=time, kwargs...)
        ModelSolution(sol, p)
    end

    # if called with vector, we transform the parameters to tuple
    function solve_model(p::Vector, V0, time, args...; kwargs...)
        solve_model(mod.θtransform(p), V0, time, args...; kwargs...)
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
    hcat([Q.(modsol.solution(t), t, modsol.θ.θflow) for t in time]...)
end


function evapotranspiration(V, t, θ)
    θ[1]*V/(V+θ[2])
end

@doc """
# Calculate evapotranspiration for each reservoir at give point in time

    evapotranspiration(modsol::ModelSolution, time)
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
