using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Plots, Distributions
enable_autocomplete_brackets(false)

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
R = [0.1 0. ; 0. 0.1]

# Initial state
global prior = MvNormal([0.,0.], [0.1 0. ; 0. 0.1])

# Storage and simulation loop
x = Vector(undef,length(data))
for t ∈ 1:length(data)
    prediction = kalman_predict(prior, A,Q)
    global prior = kalman_update(prediction, data[t], H, R)
    x[t] = prior
end


# Plot 1st dimension
#plot(0:0.1:10,[d[1] for d in data])
#plot!(0:0.1:10,[d.μ[1] for d in x],ribbon=[d.Σ[1] for d in x])

# Plot 2nd dimension
plot!(0:0.1:10,[d[2] for d in data])
plot!(0:0.1:10,[d.μ[2] for d in x],ribbon=[d.Σ[4] for d in x])

