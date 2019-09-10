using MiniFlex
using Test

import TransformVariables
import Interpolations
import ForwardDiff

@testset "Integration tests" begin

    # -----------
    # 0. fake some data

    t_rain = range(0, stop=900, length=150);
    rain_obs = [ifelse(t<500, abs(sin(t/50)*15), 0.0)  for t in t_rain]

    # function to interpolate rain observations
    rain = Interpolations.LinearInterpolation(t_rain, rain_obs,
                                              extrapolation_bc = 0)
    # must return the input for each storage at time t
    precip(t) = [rain(t), 0.0, 0.0, 0.0]

    # "observations" of Q3
    flow_data = collect(hcat([ifelse(t < 600, [t,50.0], [t, 0.0]) for t in 1:101:1000.0]...)')


    # -----------
    # 1. define model

    moddef = ModelStructure(
        # routing matrix, from column to row
        [0   0  0  0;
         0.5 0  0  0;      # 0.5*Q1 -> S2
         0.5 1  0  0;      # 0.5*Q1 + 1*Q2 -> S3
         0   0  1  0],     # Q3 -> S4

        # preciptation(t)
        precip
    )


    # -----------
    # construct model

    # construct function to solve model
    my_model = build_model(moddef)


    # define parameter tuple to test
    p = (θflow = [[0.1, 0.01],
                  [0.05, 0.01],
                  [0.02, 0.01],
                  [0.01, 0.01]],
         θevap = [[0.1, 0.01],
                  [0.05, 0.01],
                  [0.02, 0.01],
                  [0.01, 0.01]])

    # the coresponding parameter vector
    v = TransformVariables.inverse(moddef.θtransform, p)

    # solve with parameter tuple
    sol1 = my_model(p, zeros(4), 0:10.0:1000,
                    reltol=1e-3)
    # solve with parameter vector
    sol2 = my_model(v, zeros(4), 0:10.0:1000,
                    reltol=1e-3)

    t_obs = 0:50:1000
    @test isapprox(Q(sol1, t_obs), Q(sol2, t_obs), rtol=0.01)

    @test size(Q(sol1, t_obs)) == (4, length(t_obs))
    @test size(evapotranspiration(sol1, t_obs)) == (4, length(t_obs))



    # -----------
    # test compatibility with ForwarDiff

    # define a loss function
    function loss(p, t_obs, Q_obs)

        # initial bucket volume
        V_init = zeros(4)

        # solve ODE
        sol = my_model(p, V_init, t_obs)

        # get Q3
        Q3 = Q(sol, t_obs)[3,:]

        # compute loss
        sum((Q3 .- Q_obs).^2)
    end

    # derive gradient function
    loss_grad(p, t_obs, Q_obs) = ForwardDiff.gradient(p -> loss(p, t_obs, Q_obs), p)

    @test 0.0 < loss(p, flow_data[:,1], flow_data[:,2])
    @test 0.0 < loss(v, flow_data[:,1], flow_data[:,2])
    @test length(v) == length(loss_grad(v, flow_data[:,1], flow_data[:,2]))

end
