## -------------------------------------------------------
## MiniFlex.jl
##
## Andreas Scheidegger -- andreas.scheidegger@eawag.ch
## -------------------------------------------------------

using MiniFlex
using Test

import TransformVariables
import Interpolations
import ForwardDiff

@testset "Connection" begin
    c = Connection(:S1 => [:b, :a, :c])
    @test c.reservoirs[2] == [:a, :b, :c]
    c = Connection(:S1 => :out)
    @test c.reservoirs[2] == [:out]
end


@testset "Parameter dimension test" begin

    test_model = HydroModel(
        connections = [Connection(:S1 => [:S2, :S3]),
                       Connection(:S2 => :S3),
                       Connection(:S3 => :S4)],
        # preciptation(t)
        P_rate = t -> ones(4),
        PET_rate = t -> ones(4)
    )

    # wrong names
    pbad = (a = ([0.1, 0.01],   # <--
                 [0.05, 0.01],
                 [0.02, 0.01],
                 [0.01, 0.01]),
            b = ([0.1, 0.01],   # <--
                 [0.05, 0.01],
                 [0.02, 0.01],
                 [0.01, 0.01]),
            c = ([0.3, 0.7],    # <--
                 [1.0],
                 [1.0])
            )
    @test_throws ErrorException test_model(pbad, zeros(4), 0:10.0:1000)

    # routing does not sum to 1
    pbad = (θflow = ([0.1, 0.01],
                     [0.05, 0.01]),
            θevap = ([0.1, 0.01],
                     [0.05, 0.01],
                     [0.02, 0.01],
                     [0.01, 0.01]),
            θrouting = ([0.3, 0.3], # <--
                        [1.0])
            )
    @test_throws ErrorException test_model(pbad, zeros(4), 0:10.0:1000)

    # wrong parameter dims
    pbad = (θflow = ([0.1, 0.01],
                     [0.05, 0.01]), # <--
            θevap = ([0.1, 0.01],
                     [0.05, 0.01],
                     [0.02, 0.01],
                     [0.01, 0.01]),
            θrouting = ([0.3, 0.7],
                        [1.0],
                        [1.0])
            )
    @test_throws ErrorException test_model(pbad, zeros(4), 0:10.0:1000)

    pbad = (θflow = ([0.1, 0.01],
                     [0.05, 0.01]),
            θevap = ([0.1, 0.01],
                     [0.05, 0.01],
                     [0.02, 0.01],
                     [0.01, 0.01]),
            θrouting = ([0.3, 0.7, 0.2], # <--
                        [1.0],
                        [1.0])
            )
    @test_throws ErrorException test_model(pbad, zeros(4), 0:10.0:1000)

    pbad = (θflow = ([0.1, 0.01],
                     [0.05, 0.01]),
            θevap = ([0.1, 0.01],
                     [0.05, 0.01],
                     [0.02, 0.01],
                     [0.01, 0.01]),
            θrouting = ([0.3], # <--
                        [1.0],
                        [1.0])
            )
    @test_throws ErrorException test_model(pbad, zeros(4), 0:10.0:1000)

    pbad = (θflow = ([0.1, 0.01],
                     [0.05, 0.01]),
            θevap = ([0.1, 0.01],
                     [0.05, 0.01],
                     [0.02, 0.01],
                     [0.01, 0.01]),
            θrouting = ([0.3, 0.7],
                        [1.0]) # <--
            )
    @test_throws ErrorException test_model(pbad, zeros(4), 0:10.0:1000)

end

@testset "Integration tests" begin

    # -----------
    # 0. fake some data

    t_rain = range(0, stop=900, length=150);
    rain_obs = [ifelse(t<500, abs(sin(t/50)*15), 0.0)  for t in t_rain]

    # function to interpolate rain observations
    rain = Interpolations.LinearInterpolation(t_rain, rain_obs,
                                              extrapolation_bc = 0)
    # must return the input for each storage at time t
    P_rate(t) = [rain(t), 0.0, 0.0, 0.0]

    # "observations" of Q3
    flow_data = collect(hcat([ifelse(t < 600, [t,50.0], [t, 0.0]) for t in 1:101:1000.0]...)')


    # -----------
    # define model

    test_model = HydroModel(
        connections = [Connection(:S1 => [:S2, :S3]),
                       Connection(:S2 => :S3),
                       Connection(:S3 => :S4)],
        P_rate = P_rate,
        PET_rate = t -> ones(4)
    )

    # define parameter tuple to test
    p = (θflow = ([0.1, 1.1],
                  [0.1, 0.9],
                  [0.1, 1.2],
                  [0.1, 1.0]),
         θevap = ([1.0, 20.0],
                  [2.2, 20.0],
                  [3.3, 20.0],
                  [2.2, 20.0]),
         θrouting = ([0.7, 0.3],
                     [1.0],
                     [1.0])
         )


    # the coresponding parameter vector
    v = TransformVariables.inverse(test_model.θtransform, p)

    # wrong dims of initial
    @test_throws ErrorException test_model(p, zeros(2), 0:10.0:1000)

    # solve with parameter tuple
    sol1 = test_model(p, zeros(4), 0:10.0:1000,
                      reltol=1e-5)
    # solve with parameter vector
    sol2 = test_model(v, zeros(4), 0:10.0:1000,
                      reltol=1e-5)

    t_obs = 0:50:1000
    @test isapprox(Q(sol1, t_obs), Q(sol2, t_obs), rtol=0.01)

    @test size(Q(sol1, t_obs)) == (4, length(t_obs))
    @test all( Q(sol1, t_obs) .>= 0)

    # -----------
    # test compatibility with ForwarDiff

    # define a loss function
    function loss(p, t_obs, Q_obs)

        # initial bucket volume
        V_init = zeros(4)

        # solve ODE
        sol = test_model(p, V_init, t_obs)

        # get Q3
        Q3 = Q(sol, t_obs)[3,:]

        # compute loss
        sum((Q3 .- Q_obs).^2)
    end

    # derive gradient function
    loss_grad(p, t_obs, Q_obs) = ForwardDiff.gradient(p -> loss(p, t_obs, Q_obs), p)

    @test 0.0 < loss(p, flow_data[:,1], flow_data[:,2])
    @test 0.0 < loss(v, flow_data[:,1], flow_data[:,2])
    g = loss_grad(v, flow_data[:,1], flow_data[:,2])
    @test length(v) == length(g)
    @test all(abs.(g) .!= Inf)

end
