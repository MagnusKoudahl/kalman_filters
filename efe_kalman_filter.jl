using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Plots, Distributions
enable_autocomplete_brackets(false)
include("./efe_fun.jl")

# Prediction step
function kalman_predict(prior, A,Q)
    MvNormal(A * prior.μ,transpose(A) * prior.Σ * A + Q)
end

# Update step
function kalman_update(prior, y, H, R)
    v = y - H * prior.μ
    S = H * prior.Σ * H' + R
    K = prior.Σ * transpose(H) * inv(S)
    MvNormal(prior.μ + K * v, prior.Σ - K * S * transpose(K))
end

# A and H have to be I right now
# Transition matrix and process noise
A = [1. 0. ; 0. 1.]
Q = [0.1 0. ; 0. 0.1]

# Emission matrix and process noise
H = [1. 0. ; 0. 1.]
R = [1. 0. ; 0. 1.]

# Goal prior. μ only since precision is given by generative model
p_o = [10.,10.]

# Initial state
global prior = MvNormal([0.,0.], [0.1 0. ; 0. 0.1])

# Horizon
T=10

# Storage and simulation loop
q_s_vec = Vector(undef,T)
q_s_given_o_vec = Vector(undef,T)

efe_vec = Vector(undef,T)

for t ∈ 1:T
    # Calculate q(s)
    q_s = kalman_predict(prior, A,Q)
    # Calculate q(s|o)
    q_s_given_o = kalman_update(q_s, p_o, H, R)

    # Calculate EFE and store
    efe_vec[t] = efe(q_s_given_o, q_s, p_o, R)
    q_s_given_o_vec[t] = q_s_given_o
    q_s_vec[t] = q_s

    global prior = q_s
end

plot(1:T,efe_vec)
plot(1:T, [x.μ[1] for x in q_s_vec],ribbon=[x.Σ[1] for x in q_s_vec])
plot!(1:T,[x.μ[2] for x in q_s_vec],ribbon=[x.Σ[4] for x in q_s_vec])
plot!(1:T,[x.μ[1] for x in q_s_given_o_vec],ribbon=[x.Σ[1] for x in q_s_given_o_vec])
plot!(1:T,[x.μ[2] for x in q_s_given_o_vec],ribbon=[x.Σ[4] for x in q_s_given_o_vec])
