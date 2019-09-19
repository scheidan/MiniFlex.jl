## -------------------------------------------------------
## MiniFlex.jl
##
## Andreas Scheidegger -- andreas.scheidegger@eawag.ch
## -------------------------------------------------------


module MiniFlex


import Reexport
Reexport.@reexport using DifferentialEquations
using LinearAlgebra
using TransformVariables
import StaticArrays

import Base.show

export HydroModel
export Connection
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
    print(io, "Model solution for t ∈ [$tmin - $tmax]")
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
# Type for flow connection

struct Connection
    reservoirs::Pair
    fraction::Real
end
Connection(reservoirs::Pair) = Connection(reservoirs, 1)

# pretty printing short
function Base.show(io::IO, con::Connection)
    print(io, con.reservoirs[1], " → ", con.reservoirs[2], " ($(con.fraction))")
end


function routing_mat(fpaths)
    # get all reservoirs
    all_reservoirs = []
    for fp in fpaths
        push!(all_reservoirs, fp.reservoirs[1])
        push!(all_reservoirs, fp.reservoirs[2])
    end
    unique!(all_reservoirs)

    # Build adjacency Matrix
    M = zeros(length(all_reservoirs), length(all_reservoirs))
    for fp in fpaths
        from_idx = findfirst(fp.reservoirs[1] .== all_reservoirs)
        to_idx = findfirst(fp.reservoirs[2] .== all_reservoirs)
        M[to_idx, from_idx] = fp.fraction
    end
    M = M - I

    M, all_reservoirs
end



# -----------
# Type for model definition

struct HydroModel
    reservoirs::Array
    precip::Function
    θtransform::TransformVariables.AbstractTransform
    routing::AbstractMatrix
    connections::Array{Connection,1}
    dV::Function
end


function HydroModel(connections::Array{Connection,1}, precip::Function)

    # construct routing matrix
    routing, reservoirs = MiniFlex.routing_mat(connections)

    # check routing
    tot_fraction = sum(routing + I, dims=1)
    if(any(tot_fraction .> 1 + eps()))
        error("You cannot define more than 100% total outflow! Check reservoirs(s): ",
              reservoirs[tot_fraction[1,:] .> 1])
    end

    # convert routing matrix to static Array
    N = size(routing, 1)
    routing = StaticArrays.SMatrix{N,N}(routing)

    # define parameter transformation: Real vector -> NamedTuple
    θtransform = as((
        θflow = as(Vector, as(Vector, asℝ₊, 2), N),
        θevap = as(Vector, as(Vector, asℝ₊, 2), N)
    ))

    # define ODE for all storages (p is a NamedTuple)
    function dV(dV,V,p,t)
        # calculate flows
        outQ = Q.(V, t, p.θflow)
        dV .= precip(t) .+ routing*Q.(V, t, p.θflow)
        # substract evapotranspiration
        dV .-= evapotranspiration.(V, t, p.θevap)
    end

    HydroModel(reservoirs, precip, θtransform, routing, connections, dV)
end


@doc """
# Define and run a hydrological model

## Model definition

  HydroModel(connections::Array{Connection,1}, precip::Function)

where
- `connections::Array`: array of `Connection`s to define reservoir connections
- `precip::Function`: function that returns a vector of rain intensities
                      for  each reservoir at any given time point. Takes time as argument.


### Example of routing

We have four reservoirs S1, S2, S3, S4 (any other names could be used):
```julia
[Connection(:S1 => :S2, 0.6),  # 60% off S1 flows to S2
 Connection(:S2 => :S3, 0.4),
 Connection(:S1 => :S3),       # default is 100%
 Connection(:S3 => :S4)]
```

## Run a HydroModel

Objects of type `HydroModel` are callable to run the simulation:

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
    [Connection(:S1 => :S2, 0.6),   # the names :S1 and :S2 are arbitrary
     Connection(:S2 => :S3, 0.4),
     Connection(:S1 => :S3),
     Connection(:S3 => :S4)],
    ## precipitation(t)
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

# pretty printing verbose
function Base.show(io::IO, ::MIME"text/plain", m::HydroModel)

    println(io, "HydroModel model with $(size(m.routing,1)) reservoirs connected by:")
    for f in m.connections
        println(io, " ", f)
    end
    println(io, "\nThe parameters must be ordered as:")
    for (i, r) in enumerate(m.reservoirs)
        i==1 ? print(io, " $i: ", r) : print(io, ", $i: ", r)
    end
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
