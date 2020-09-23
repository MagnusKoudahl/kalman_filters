using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Plots, Distributions, ForwardDiff, LinearAlgebra
enable_autocomplete_brackets(false)

# shorthand
jacobian = ForwardDiff.jacobian
gradient = ForwardDiff.gradient

# Generate data from correct model
function generate_data(start,f,h,σ)
    states = Vector(undef,T+1)
    obs = Vector(undef,T)
    states[1] = start
    for t in 1:T
	states[t+1] = f(states[t]) # No noise since we just want observations
	obs[t] = h(states[t])
    end
    noise = [(rand(size(o)[1]) .- 0.5) .* σ for o in obs]
    return obs .+ noise,states
end

# Transition function
function f(x)
    # Returns [f(x),0..]
    #g = 0.1
    [x[1] + x[2] * Δt, x[2] - 1.0 * sin(x[1]) * Δt,0,0]
end

# Emission function
function h(x)
    x
end


# Prediction step
function kalman_predict(prior, f, Q)
    ∇f = jacobian(f,mean(prior))
    Σ = round.(∇f * cov(prior) * transpose(∇f) + Q, digits=5) # Round to avoid issues with numerical precision
    MvNormal(f(mean(prior)),Σ)
end

# Update step
function kalman_update(prior, y, h, R)
    v = y - h(mean(prior))
    ∇h = jacobian(h,mean(prior)) # returns the transpose for some reason?
    S = ∇h' * cov(prior) * ∇h .+ R# So transposes are reversed
    K = cov(prior) * ∇h * inv(S)
    Σ = round.(cov(prior) - K * S * transpose(K),digits=5) # Round to avoid numerical errors
    MvNormal(mean(prior) + K * v, Σ)
end

# Params
Δt = 0.01
start = [1.,1.]
T = 1000
σ = [1.0,1.0]
# Process noise
Q = [.1 0. ; 0. .1]
# Emission noise
R = [σ[1] 0.0 ; 0.0 σ[2]]
# Initial state
global prior = MvNormal(start, [1.0 0. ; 0. 1.0])
#data = generate_data(start,T,g,Δt,σ)
data, states = generate_data(start,f,h,σ)

# Storage and simulation loop
x = Vector(undef,length(data))
for t ∈ 1:length(data)
    prediction = kalman_predict(prior, f, Q)
    global prior = kalman_update(prediction, data[t], h, R)
    x[t] = prior
end


# Plots
plot([d[1] for d in data],label="Data 1")
plot!([d[2] for d in data],label="Data 2")

plot!([d[1] for d in states],label="states 1")
plot!([d[2] for d in states],label="states 2")

plot!([d.μ[1] for d in x],ribbon=[d.Σ[1] for d in x], label="Filter 1")
plot!([d.μ[2] for d in x],ribbon=[d.Σ[4] for d in x], label="Filter 2")
