using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Plots, Distributions
enable_autocomplete_brackets(false)
include("./efe_fun_2.jl")

# Generate data from a noisy sin function
function generate_sin_data(minimum,maximum,stepsize,σ)
    [sin.([x;x]) + rand(MvNormal([0.;0.],σ)) for x in minimum:stepsize:maximum]
end
data = generate_sin_data(0,10,0.1,0.1)

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

# Transition matrix and process noise
A = [1. 0. ; 0. 1.]
Q = [0.1 0. ; 0. 0.1]

# Emission matrix and process noise
H = [1. 0. ; 0. 1.]
R = [1.1 0. ; 0. 1.1]

# Goal prior
p_o= MvNormal([10.,10.],[0.1 0. ; 0. 0.1])

# Initial state
#global prior_q = MvNormal([0.,0.], [0.1 0. ; 0. 0.1])
global prior = MvNormal([0.,0.], [0.1 0. ; 0. 0.1])

# Horizon
T=10

# Storage and simulation loop
x = Vector(undef,length(data))
q_s = Vector(undef,T)
q_s_given_o = Vector(undef,T)

efe_vec = Vector(undef,T)

for t ∈ 1:T
    prediction = kalman_predict(prior, A,Q)
    q_s[t] = prediction
    global prior = prediction
    efe_vec[t] = efe(p_o,prediction,R)
    #global prior = kalman_update(prediction, data[t], H, R)
    #x[t] = prior
end
efe_vec
