using Pkg;Pkg.activate(".");Pkg.instantiate()
using Distributions, PDMats, OhMyREPL, Zygote
enable_autocomplete_brackets(false)
include("utils.jl")

function efe(q_s_t_min,p_s_t_min,p_o_given_m,π_t,A,Q,H,R)

    # Compute next state q(s_t|s_{t-1}, π_t). Known dynamics A, so only param is action π_t
    q_s_t = MvNormal(A*q_s_t_min.μ .+ π_t,A*q_s_t_min.Σ*transpose(A) + Q)

    # Compute joint q(o,s | π_t)
    q_os_t = MvNormal([q_s_t.μ;H*q_s_t.μ],
		      [q_s_t.Σ q_s_t.Σ*transpose(H) ;
		       H*q_s_t.Σ H*q_s_t.Σ*transpose(H) + R])

    # Compute prior p(s_t | s_{t-1}). p(π_t) = 0, ie we don't assume any action apriori
    p_s_t = MvNormal(A*p_s_t_min.μ ,A*p_s_t_min.Σ*transpose(A) + Q)

    # Compute joint p(o,s|s_{t-1{)
    p_os_t = MvNormal([p_s_t.μ;H*p_s_t.μ],
		      [p_s_t.Σ p_s_t.Σ*transpose(H) ;
		       H*p_s_t.Σ H*p_s_t.Σ*transpose(H) + R])

    # Compute marginal p(o_t)
    p_o = MvNormal(H*p_s_t.μ,H*p_s_t.Σ*transpose(H) + R)

    # Evaluate loss as H[q(o,s|pi) || p(o,s|s_{t-1})] - H[q(s_t)] + H[p(o) || p(o|m)]
    loss = cross_entropy_gaussians(q_os_t,p_os_t) - entropy(q_s_t) + cross_entropy_gaussians(p_o,p_o_given_m)
end


# Transition matrix and process noise
A = [1. 0. ; 0. 1.]
Q = [0.1 0. ; 0. 0.1]

# Emission matrix and process noise
H = [1. 0. ; 0. 1.]
R = [1. 0. ; 0. 1.]

# Goal prior.
p_o_given_m = MvNormal([2.,2.],[1.0 0.0 ; 0.0 1.0])
q_s_t_min = MvNormal([1.;1.],[1. 0. ;0. 1.])
p_s_t_min = MvNormal([1.;1.],[1. 0. ;0. 1.])

# Action as a scalar param. Needs extension to distribution
π_t = [1.,1.]

# Horizon
T = 2
efe_vec = Vector(undef,T)
for t ∈ 1:T
    # Compute EFE
    efe_vec[t] = efe(q_s_t_min,p_s_t_min,p_o_given_m,π_t,A,Q,H,R)

    # Update distributions
    global q_s_t_min = MvNormal(A*q_s_t_min.μ .+ π_t,A*q_s_t_min.Σ*transpose(A) + Q)

    global p_s_t_min = MvNormal(A*p_s_t_min.μ ,A*p_s_t_min.Σ*transpose(A) + Q)

end



