using Distributions, StatsBase, LinearAlgebra, PDMats

function kl_univariate_gaussians(p,q)
    # KL of 2 univariate Normals
    log(q.σ / p.σ) + (p.σ + (p.μ - q.μ)^2)/ (2*q.σ) - 0.5
end

function kl_mvgaussians(p,q)
    # Fix division by 0 and make sure dim of n is correct. Also having to use "Matrix()" is bad. In general this function is fugly
    0.5* (
	  log(det(Matrix(q.Σ)) / det(Matrix(p.Σ))) -
	  size(p)[1] +
	  tr(q.Σ \ Matrix(p.Σ)) +
	  transpose(q.μ - p.μ) * (q.Σ \ (q.μ - p.μ))
	  )
end

function cross_entropy_gaussians(p,q)
    # Calculates cross entropy of 2 MvNormals
    0.5 * log(2*π) + logdet(q.Σ) + invquad(q.Σ,q.μ) - transpose(p.μ)*(q.Σ \ q.μ) - transpose(q.μ)* (q.Σ \ p.μ) + tr(q.Σ \ (p.μ * transpose(p.μ) + p.Σ))
end

function joint(prior,likelihood)
    # Creates a joint out of prior and likelihood for MvNormals with direct mapping
    Λ = prior.Σ.mat
    L = likelihood.Σ.mat
    MvNormal(cat(prior.μ,prior.μ,dims=1),[Λ Λ ; Λ (Λ + L)])
end

function posterior(prior,likelihood)
    # Computes a posterior for gaussians given a prior and a likelihood. Assumes that observation was equal to the mean since it's for use with goal priors
    μ = prior.μ
    x = likelihood.μ
    Σ_μ = prior.Σ.mat
    Σ_x = likelihood.Σ.mat

    MvNormal(Σ_μ * inv(Σ_μ + Σ_x) * x + Σ_x * inv(Σ_μ + Σ_x) * μ, Σ_μ * inv(Σ_μ + Σ_x) * Σ_x)
end

