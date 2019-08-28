using MiniFlex
using Test

@testset "Integration test" begin

    import Interpolations
    import StaticArrays



    # -----------
    # 0. fake some data

    t_rain = range(0, stop=900, length=150);
    rain_obs = [ifelse(t<500, abs(sin(t/50)*15), 0.0)  for t in t_rain]

    # function to interpolate rain obersvations
    rain = Interpolations.LinearInterpolation(t_rain, rain_obs,
                                              extrapolation_bc = 0)
    # must return the input for each storage at time t
    precip(t) = [rain(t), 0.0, 0.0, 0.0]

    # -----------
    # 1. define model

    # from column to row
    M = StaticArrays.@SMatrix [0   0  0  0;
                               0.5 0  0  0;      # 0.5*Q1 -> S2
                               0.5  1 0  0;      # 0.5*Q1 + 1*Q2 -> S3
                               0    0  1 0]      # Q3 -> S4


    moddef = ModelStructure(
        M,                          # routing matrix
        precip                      # preciptation(t)
    )


    # -----------
    # construct model

    # construct function to solve model
    my_model = build_model(moddef)


    # define parameters to test
    p = Param(
        FlowParam([0.1, 0.01],    # θ for Q1
                  [0.05, 0.01],    # θ for Q2
                  [0.02, 0.01],     # θ for Q3
                  [0.013, 0.01]),   # θ for Q4
        EvapoParam([0.0, 1.0],    # θ for evapo1
                   [0.00, 1.0],    # θ for evapo2
                   [0.00, 1.0],    # θ for evapo3
                   [0.00, 1.0])    # θ for evapo4
    )


    # solve with V0=zeros(4)
    sol = my_model(p, zeros(4), 0:10.0:1000,
                   reltol=1e-3);

    t_obs = 0:50:1000
    @test length(Q(sol, t_obs)) == length(t_obs)
    @test length(evapotranspiration(sol, t_obs)) == length(t_obs)


end


@testset "Forward diff" begin

    import ForwardDiff


    flow_data = collect(hcat([ifelse(t < 600, [t,50.0], [t, 0.0]) for t in 1:101:1000.0]...)')

    # loss function
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



    @test loss(p, flow_data[:,1], flow_data[:,2])
    @test length(p) == length(loss_grad(p, flow_data[:,1], flow_data[:,2]))

end