using StochOpt
using Match


function solve_logreg(X::Matrix{Float64}, y::Vector{Float64}, lambda::Float64,
                      n_iter::Int64, method_name::AbstractString = "SVRG", batch_size::Int64 = 100)

    # Set option for StochOpt solvers
    options = set_options(
        max_iter=n_iter,  # run n_iter epochs of the considered function
        batchsize=batch_size,
        embeddim=0,
        printiters=false,
        stepsize_multiplier=1.0,
        exacterror=false,
    );

    # In StochOpt, the objective is a mean logistic loss compared to the sum in
    # benchopt, so we need to divide lambda per the number of sample. We cannot
    # use regularizor_parameter as this only work when lambda == -1.
    n_samples = size(X, 2);
    lambda /= n_samples;

    # Load logistic problem with the given matrices
    prob = load_logistic_from_matrices(
        X, y, "benchopt", options, lambda=lambda, scaling="none"
    );


    if method_name in ["SVRG_bubeck", "Free_SVRG", "Leap_SVRG"]

        # This gives the theoretical step size for convergence of the methods, which is not
        # available for batchsize != 1
        options.batchsize = 1;
        options.stepsize_multiplier = -1.0;
        # sampling strategy for the stochastic estimates
        sampling = StochOpt.build_sampling("nice", prob.numdata, options);
    end

    @match method_name begin
        "SVRG_bubeck"   => (method = StochOpt.initiate_SVRG_bubeck(
            prob, options, sampling, numinneriters=-1
        ))
        "Free_SVRG"     => (method = StochOpt.initiate_Free_SVRG(
            prob, options, sampling, numinneriters=prob.numdata, averaged_reference_point=true
        ))
        "Leap_SVRG"     => (method = StochOpt.initiate_Leap_SVRG(
            prob, options, sampling, 1/prob.numdata
        ))
        "L_SVRG_D"      => (method = StochOpt.initiate_L_SVRG_D(
            prob, options, sampling, 1/prob.numdata
        ))
        "SAGA_nice"     => (method = StochOpt.initiate_SAGA_nice(prob, options))
        _               => (method = method_name)
    end #

    prob.fsol = 0.0  # disable loading of the solution
    return minimize_func(prob, method, options);

end


# Different naming with minimizeFunc which already exists inside StochOpt
function minimize_func(prob::Prob, method_input, options::MyOptions;)
    if options.initial_point == "randn" # set initial point
        x = randn(prob.numfeatures);
    elseif(options.initial_point == "rand")
        x = rand(prob.numfeatures); #
    elseif(options.initial_point == "ones")
        x = ones(prob.numfeatures); #
    else
        x = zeros(prob.numfeatures); #
    end

    if typeof(method_input) == String
        method = StochOpt.boot_method(method_input, prob, options);
        if method=="METHOD DOES NOT EXIST"
            println("FAIL: unknown method name:")
            return
        end
    else
        method = method_input;
        method.bootmethod(prob, method, options);
    end

    d = zeros(prob.numfeatures); # Search direction vector

    for iter = 1:options.max_iter
        ## Taking a step
        method.stepmethod(x, prob, options, method, iter, d) # mutating function
        x[:] = x + method.stepsize * d
    end # End of For loop

    return x
end
