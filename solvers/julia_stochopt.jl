using StochOpt


function solve_logreg_svrg(X, y, lambda, n_iter)
    options = set_options(max_iter=n_iter, max_time=350.0, max_epocs=n_iter,
                          exacterror=false, skip_error_calculation=n_iter+1,
                          repeat_stepsize_calculation=true, rep_number=4,
                          batchsize=100, embeddim=10, printiters=false);

    probname = "empty"  # just a placeholder for the load logistic func below
    prob = load_logistic_from_matrices(X, y, probname, options, lambda=lambda);
    prob.fsol = 0.0  # disable loading of the solution
    output = minimizeFunc(prob, "SVRG", options);

    # return the fitted coefficients
    return output.xfinal 

end


function solve_logreg_saga_batch(X, y, lambda, n_iter)
    options = set_options(max_iter=n_iter, max_time=350.0, max_epocs=n_iter,
                          repeat_stepsize_calculation=true, rep_number=4,
                          batchsize=100, embeddim=10, printiters=false);

    probname = "empty"  # just a placeholder for the load logistic func below
    prob = load_logistic_from_matrices(X, y, probname, options, lambda=lambda);
    prob.fsol = 0.0  # disable loading of the solution
    output = minimizeFunc(prob, "SAGA_nice", options);

    # return the fitted coefficients
    return output.xfinal 
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
        method = boot_method(method_input, prob, options);
        if method=="METHOD DOES NOT EXIST"
            println("FAIL: unknown method name:")
            return
        end
    else
        method = method_input
        method.bootmethod(prob, method, options)
    end

    d = zeros(prob.numfeatures); # Search direction vector

    for iter = 1:options.max_iter
        ## Taking a step
        method.stepmethod(x, prob, options, method, iter, d) # mutating function
        x[:] = x + method.stepsize * d
    end # End of For loop

    return x
end
