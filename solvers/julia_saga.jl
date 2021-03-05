using Core
using LinearAlgebra
using StatsBase # sample function

# TODO : import shared functions from file
# include("./benchmarks/logreg_l2/solvers/logistic.jl")


# Gradient evaluation
function sigmoid(z::Array{Float64})
    # This function computes the sigmoid function:
    # \sigma(z) = 1 / (1 + e^(-z)) .
    # Let the i-th loss be
    # \phi_i (z) = \log \left( 1 + e^{-y_i z} \right) .
    # Then its derivative is
    # \phi_i^' (z) = -y_i \sigma(-y_i z)
    idx = z .> 0
    out = zeros(size(z))
    out[idx] = (1 .+ exp.(-z[idx])).^(-1)
    exp_t = exp.(z[.~idx])
    out[.~idx] = exp_t ./ (1. .+ exp_t)
    return out
end

function logreg_l2_Jac!(X, y, w::Array{Float64}, lambda::Float64, B::Array{Int64}, Jac::Array{Float64})
    n_samples = size(X, 1)
    z = sigmoid(y[B] .* (X[B, :]*w));
    Jac[:, B] = n_samples .* X[B, :]' .* (y[B] .* (z .- 1))' .+ (lambda .* w); # J_{:i}^{t+1} <- \nabla f_i (w^t)
end


function solve_logreg_l2(X, y, lambda::Float64, n_iter::Int64; batch_size::Int64=1, unbiased::Bool=false)
    """
    Implementation based on Algorithm 2 of
    N. Gazagnadou, R. M. Gower, J. Salmon, `Optimal Mini-Batch and Step Sizes for SAGA`, ICML 2019.
    """
    # TODO : use expected smoothness instead -> larger step size with minibatching

    n_samples = size(X, 1)
    Lmax = (n_samples/4) * maximum(sum(X .^ 2, dims=2)) + lambda
    step_size = 1. / Lmax
    println("Step size SAGA = ", step_size, "\n")

    n_features = size(X, 2)
    w = zeros(n_features, 1)

    Jac = zeros(n_features, n_samples) # Jacobian estimate
    # logreg_l2_Jac!(X, y, w, lambda, collect(1:n_samples), Jac) # full gradient Jacobian init # FIXME
    aux = zeros(n_features, 1) # auxiliary vector
    grad_estim = zeros(n_features, 1) # stochastic gradient estimate, SAGA if unbiased = true, SAG else
    u = sum(Jac, dims=2) # SAG (biased) estimate
    for i ∈ 1:n_iter
        B = sample(1:n_samples, batch_size, replace=false) # sampling a mini-batch

        # Assign each gradient to a different column of Jac
        aux[:] = -sum(Jac[:, B], dims=2) # Calculating the auxiliary vector
        logreg_l2_Jac!(X, y, w, lambda, B, Jac) # Update of the Jacobian estimate
        aux[:] += sum(Jac[:, B], dims=2) # aux = \sum_{i \in B} (\nabla f_i (w^t) - J_{:i}^t)

        # Update of the unbiased gradient estimate: g^k
        if unbiased
            grad_estim[:] = u .+ ((1. / length(B)) .* aux) # SAGA unbiased descent direction
        else
            grad_estim[:] = u; # SAG biased descent direction
        end

        # Update SAG biased estimate: 1/n J^{k+1}1 = 1/n J^k 1 + 1/n (DF^k-J^k) Proj 1
        u[:] = u .+ ((1. / n_samples) .* aux)

        # Update the vector of weights through a stochastic step
        w -= step_size .* grad_estim
    end
    return w
end
