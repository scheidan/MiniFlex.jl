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

using RecipesBase


import Base.show

export HydroModel
export Connection
export Q, ET


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
    reservoir_names::Array{Symbol}
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
    println(io, " θrouting:")
    println(io, "   $(sol.θ.θrouting)")

    println(io, "\nFlows time series can be obtained with:\n"
            * " Q(solution, time_range)")
end


# -----------
# Type for flow connection

struct Connection
    reservoirs::Pair{Symbol, Vector{Symbol}}
    function Connection(r::Pair{Symbol, Vector{Symbol}})
        sort!(r[2])
        new(r)
    end
end

Connection(r::Pair{Symbol, Symbol}) = Connection(r[1] => [r[2]])


# pretty printing short
function Base.show(io::IO, con::Connection)
    print(io, con.reservoirs[1], " → ")
    for (i,s) in enumerate(con.reservoirs[2])
        i == 1 ? print(s) : print(", ", s)
    end
end


function routing_mat(fpaths)
    # get all reservoirs
    all_reservoirs = Symbol[]
    for fp in fpaths
        push!(all_reservoirs, fp.reservoirs[1])
        append!(all_reservoirs, fp.reservoirs[2])
    end
    sort!(unique!(all_reservoirs))

    # Build adjacency Matrix
    n = length(all_reservoirs)
    M = zeros(Bool, n, n)
    for fp in fpaths
        for i in 1:length(fp.reservoirs[2])
            from_idx = findfirst(fp.reservoirs[1] .== all_reservoirs)
            to_idx = findfirst(fp.reservoirs[2][i] .== all_reservoirs)
            M[to_idx, from_idx] = true
        end
    end

    M, all_reservoirs
end



# -----------
# Type for model definition

struct HydroModel
    reservoir_names::Array{Symbol}
    P_rate::Function
    PET_rate::Function
    θtransform::TransformVariables.AbstractTransform
    routing_mask::BitArray{2}
    connections::Array{Connection,1}
    dV::Function
end


function HydroModel(; connections::Array{Connection,1}, P_rate::Function, PET_rate::Function)

    # construct mask routing matrix
    routing_mask, reservoir_names = routing_mat(connections)
    N = size(routing_mask, 1)

    # connections
    sort!(connections, lt=(a,b) -> a.reservoirs[1] < b.reservoirs[1])
    Nout = [length(c.reservoirs[2]) for c in connections] # number of outgoing connections

    # define parameter transformation: Real vector -> NamedTuple
    θtransform = as((
        θflow = as(Tuple(as(Vector, asℝ₊, 2) for _ in 1:N)),
        θevap = as(Tuple(as(Vector, asℝ₊, 2) for _ in 1:N)),
        θrouting = as(Tuple(UnitSimplex(n) for n in Nout))
    ))

    # define ODE for all storages (p is a NamedTuple)
    function dV(dV, V, p, t, routing, outQ)
        # calculate flows
        outQ .= Q.(V, t, p.θflow)
        dV .= routing*outQ
        # add preciptation
        dV .+= P_rate(t)
        # substract evapotranspiration
        dV .-= ET.(V, PET_rate(t), t, p.θevap)
    end

    HydroModel(reservoir_names, P_rate, PET_rate, θtransform, routing_mask, connections, dV)
end


@doc """
# Define and run a hydrological model

## Model definition

  HydroModel(connections::Array{Connection,1}, P_rate::Function, PET_rate::Function)

where
- `connections::Array`: array of `Connection`s to define reservoir connections
- `P_rate::Function`:   function that returns a vector of rain intensities
                        for each reservoir at any given time point. Takes time as argument.
                        Note, reservoirs are alphabetically sorted.
- `PEV_rate::Function`: function that returns a vector of the potential evapotranspiration (PET)
                        for each reservoir at any given time point. Takes time as argument.
                        Note, reservoirs are alphabetically sorted.


### Example of routing

We have four reservoirs S1, S2, S3, S4 (any other names could be used):
```julia
[Connection(:S1 => [:S2, :S3]),
 Connection(:S2 => [:S3, :S5]),
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
    connections = [Connection(:S2 => :S3),
                   Connection(:S1 => [:S3, :S2]),
                   Connection(:S3 => :S4)],
    P_rate = P_rate,        # precipitation
    PET_rate = PET_rate     # potential evapotranspiration
)

## 2) run model

p = (θflow = ([0.1, 1.1],
              [0.1, 0.9],
              [0.1, 1.2],
              [0.1, 1.0]),
     θevap = ([1.0, 20.0],
              [2.2, 20.0],
              [3.3, 20.0],
              [2.2, 20.0]),
     θrouting = ([0.3, 0.7], # connection from :S1
                 [1.0],      # connection from :S2
                 [1.0])      # connection from :S3
     )

V0 = zeros(4)   # inital storage

sol = my_model(p, V0, 0:10.0:1000) # default solver
sol2 = my_model(randn(17), V0, 0:10.0:1000) # call with parameter vector
sol3 = my_model(p, V0, 0:10.0:1000, ImplicitMidpoint(), reltol=1e-3, dt=5)

# requires`Plots` to by loaded
plot(sol, value="Q")
plot(sol, value="volume") # requires

# extract run-off time-series (note, time points can be different)
t_obs = 10:2.5:100
Q(sol, t_obs)
```
"""
function (m::HydroModel)(p::NamedTuple, V0, time, args...; kwargs...)

    # check parameters
    n_storages = size(m.routing_mask, 1)

    for k in [:θflow, :θevap, :θrouting]
        if  !(k in keys(p))
            error("Parameters must contain a field '$(k)'")
        end
    end

    if(!(length(p.θflow) == length(p.θevap) == n_storages))
        error("Parameter dimensions are not matching the number " *
              "of storages ($(n_storages))!")
    end

    if(length(m.P_rate(minimum(time))) != n_storages)
        error("Precipitation function must return a vector of the same length " *
              "as the number of storages ($(n_storages))!")
    end

    if(length(V0) != n_storages)
        error("Initial values must be a vector of the same length " *
              "as the number of storages ($(n_storages))!")
    end

    if(length(p.θrouting) != length(m.connections))
        error("Routing parameters (θrouting):", ("\n - $(p)" for p in p.θrouting)...,
              "\ndo not match connections:", ("\n - $(c)" for c in m.connections)..., "\n")
    end

    for i in 1:length(p.θrouting)
        if length(p.θrouting[i]) != length(m.connections[i].reservoirs[2])
            error("Parameters ($(p.θrouting[i])) do not match connection:\n",
                  m.connections[i])
        end
        if sum(p.θrouting[i]) ≉ 1
            error("Parameters ($(p.θrouting[i])) do not sum to one, see connection:\n",
                  m.connections[i])
        end
    end

    # construct routing matrix
    routing = zeros(eltype(p.θflow[1]), n_storages, n_storages) - I
    for i in 1:length(p.θrouting)
        routing[m.routing_mask[:,i], i] .= p.θrouting[i]
    end

    # solve ode
    outQ = zeros(eltype(p.θflow[1]), n_storages)
    prob = ODEProblem((dV,V,p,t) -> m.dV(dV,V,p,t,routing, outQ),
                      nested_eltype(p.θflow).(V0),
                      (minimum(time), maximum(time)),
                      p)
    sol = solve(prob, args...; saveat=time, kwargs...)

    ModelSolution(sol, p, m.reservoir_names)
end

function (m::HydroModel)(p::AbstractArray, V0, time, args...; kwargs...)
    # if called with vector, we transform the parameters to tuple
    m(m.θtransform(p), V0, time, args...; kwargs...)
end


# pretty printing short
function Base.show(io::IO, m::HydroModel)
    print(io, "HydroModel ($(size(m.routing_mask,1)) reservoirs)")
end

# pretty printing verbose
function Base.show(io::IO, ::MIME"text/plain", m::HydroModel)

    println(io, "HydroModel model with $(size(m.routing_mask,1)) reservoirs connected by:")
    for f in m.connections
        println(io, " ", f)
    end
    println(io, "\nParameters and outputs are ordered as:")
    for (i, r) in enumerate(m.reservoir_names)
        i==1 ? print(io, " $i: ", r) : print(io, ", $i: ", r)
    end
end


# -----------
# constitutive functions for flow and evapotranspiration

# see:.
# Fenicia, F., Kavetski, D., Savenije, H.H.G., Clark, M.P.,
# Schoups, G., Pfister, L., Freer, J., 2014. Catchment properties,
# function, and conceptual model representation: is there a
# correspondence? Hydrological Processes 28,
# 2451–2467. https://doi.org/10.1002/hyp.9726

function Q(V, t, θ)
    V <= 0 ? zero(V) : θ[1] * V^θ[2]
end

@doc """
# Calculate outflow from each reservoir for give point in time

    Q(sol::ModelSolution, time)
"""
function Q(modsol::ModelSolution, time)
    hcat([Q.(modsol.solution(t), t, modsol.θ.θflow) for t in time]...)
end


function ET(V, PET, t, θ)
    V <= 0 ? zero(V) : θ[1] * PET * (1 - exp(-V/θ[2]))
end


# -----------
# helpers

function nested_eltype(x)
    y = eltype(x)
    while y <: Union{AbstractArray, Tuple}
        y = eltype(y)
    end
    return(y)
end


# -----------
# Plot recipes


@recipe function f(::Type{ModelSolution}, modsol::ModelSolution; value="Q", add_marker=false)
    xlabel --> "time"
    linewidth --> 2
    if add_marker
        markershape --> :circle
        markersize --> 1
    end
    if value == "Q"
        seriestype  :=  :path
        label --> ["outflow of $r" for r in modsol.reservoir_names]
        ylab --> "flow"
        Q(modsol, modsol.solution.t)'
    elseif value == "volume"
        label --> ["volume of $r" for r in modsol.reservoir_names]
        ylab --> "volume"
        modsol.solution
    else
        error("For 'value' only the strings \"Q\" or \"volume\" are allowed!")
    end
end



end # module
