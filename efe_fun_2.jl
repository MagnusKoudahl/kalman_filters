#using Pkg;Pkg.activate(".");Pkg.instantiate()
using LinearAlgebra,Distributions,PDMats
# Calculate EFE from the original decomposition in Friston 2015.
# EFE 	= E_{q(s)}[ G - H[q(s)]]
# 	= E_{q(s)}[ -E_{p(o|s)}log p(o,s) - H[q(s)]]
# 	= H[q_tilde(o,s), p(o,s)] - H[q(s)]]
# Where the first term is a crossentropy. All terms can be solved analytically for linear gaussian systems, giving an analytical solution to EFE in continuous time


# Calculate EFE. Takes p(o), q(s) and likelihood variance of p(o|s) as input
function efe(p_o,q_s,Σ_p_o_given_s)

    # Compute the posterior q(s | o) after observing mean of p_o
    q_s_given_o = MvNormal(p_o.Σ.mat * inv(p_o.Σ.mat + q_s.Σ.mat) * q_s.μ + q_s.Σ.mat * inv(p_o.Σ.mat + q_s.Σ.mat) * p_o.μ, p_o.Σ.mat * inv(p_o.Σ.mat + q_s.Σ.mat) * q_s.Σ.mat)

    # Compute q_tilde as p(o|s)q(s)
    q_tilde = MvNormal(cat(q_s.μ,q_s.μ,dims=1),
		       [q_s.Σ.mat q_s.Σ.mat ;
			q_s.Σ.mat Σ_p_o_given_s + q_s.Σ.mat])

    # Compute p(o,s) as q(s|o)p(o)
    p_os = MvNormal(cat(p_o.μ,p_o.μ,dims=1),
		    [p_o.Σ.mat p_o.Σ.mat ;
		     p_o.Σ.mat q_s_given_o.Σ + p_o.Σ.mat])

    # Energy as crossentropy of q_tilde and p_os
    energy = 0.5 * log(2*π) + logdet(q_tilde.Σ) + invquad(q_tilde.Σ,q_tilde.μ) - transpose(p_os.μ)*(q_tilde.Σ \ q_tilde.μ) - transpose(q_tilde.μ)* (q_tilde.Σ \ p_os.μ) + tr(q_tilde.Σ \ (p_os.μ * transpose(p_os.μ) + p_os.Σ))

    # EFE as energy - entropy
    expected_fe = energy - entropy(q_s)
end
