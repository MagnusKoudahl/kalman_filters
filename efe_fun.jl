#using Pkg;Pkg.activate(".");Pkg.instantiate()
using LinearAlgebra,Distributions,PDMats
# Calculate EFE from the original decomposition in Friston 2015.
# EFE 	= E_{q(s)}[ G - H[q(s)]]
# 	= E_{q(s)}[ -E_{p(o|s)}log p(o,s) - H[q(s)]]
# 	= H[q_tilde(o,s), p(o,s)] - H[q(s)]]
# Where the first term is a crossentropy. All terms can be solved analytically for linear gaussian systems, giving an analytical solution to EFE in continuous time


# Calculate EFE.
function efe(q_s_given_o,q_s,p_o,Σ_p_o_given_s)
    # q_s_given_o = q(s|o), can be obtained from Kalman update step
    # q_s = q(s), can be obtained from Kalman prediction step
    # p_o = p(o|m), goal prior mean. We only supply the mean, precision is calculated from generative model
    # Σ_p_o_given_s = R, likelihood variance. Known from model specification
# This implementation assumes H = A = I, ie all transition models are just the identity matrix

    # Compute q_tilde as p(o|s)q(s). This does not use the goal prior
    q_tilde = MvNormal(cat(q_s.μ,q_s.μ,dims=1),
		       [q_s.Σ.mat q_s.Σ.mat ;
			q_s.Σ.mat Σ_p_o_given_s + q_s.Σ.mat])


    # Compute p(o,s) as q(s|o)p(o). This uses the goal prior. Currently relies heavily on H = inv(H) = I
    p_os = MvNormal([p_o;p_o],
		    [q_s_given_o.Σ.mat + Σ_p_o_given_s q_s_given_o.Σ.mat + Σ_p_o_given_s ;
		     q_s_given_o.Σ.mat + Σ_p_o_given_s q_s_given_o.Σ + q_s_given_o.Σ.mat + Σ_p_o_given_s])

    # Energy as crossentropy of q_tilde and p_os
    energy = 0.5 * log(2*π) + logdet(q_tilde.Σ) + invquad(q_tilde.Σ,q_tilde.μ) - transpose(p_os.μ)*(q_tilde.Σ \ q_tilde.μ) - transpose(q_tilde.μ)* (q_tilde.Σ \ p_os.μ) + tr(q_tilde.Σ \ (p_os.μ * transpose(p_os.μ) + p_os.Σ))

    # EFE as energy - entropy
    expected_fe = energy - entropy(q_s)
end
