using StochOpt


function solve_logreg_svrg(X, y, lambda, n_iter)
    options = set_options(max_iter=n_iter, max_time=350.0, max_epocs=30,
                          repeat_stepsize_calculation=true, rep_number=4,
                          batchsize=100, embeddim=10, printiters=false);

    probname = "empty"  # just a placeholder for the load logistic func below
    prob = load_logistic_from_matrices(X, y, probname, options, lambda=lambda);
    prob.fsol = 0.0  # disable loading of the solution
    output = minimizeFunc(prob, "SVRG", options);

    # return the fitted coefficients
    return output.xfinal 

end
