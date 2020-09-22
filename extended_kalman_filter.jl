# Work in progress. Check data generation and fix gradient errors in emission function. Probably also some other stuff
using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Plots, Distributions, ForwardDiff, LinearAlgebra
enable_autocomplete_brackets(false)

# shorthand
jacobian = ForwardDiff.jacobian
gradient = ForwardDiff.gradient

# Generate data from correct model
function generate_data(start,T,g,Δt,σ)
    states = Vector(undef,T+1)
    obs = Vector(undef,T)
    states[1] = start
    for t in 1:T
	states[t+1] = f(vcat(states[t],Δt,g)) #.+ rand(size(σ),σ)
	obs[t] = h(states[t])
    end
    return obs
end

# Transition function
function f(x)
    # x[1:2] = States
    # x[3] = Δt
    # x[4] = g
    [x[1] + x[2] * x[3],x[2] - x[4] * sin(x[1]) * x[3]]
end

# Emission function
function h(x)
    # x = states
    vcat(sin(x[1]),0.)
end


# Prediction step
function kalman_predict(prior, f, Q)
    # Slightly hacky jacobian.
    ∇f = jacobian(f,vcat(mean(prior),Δt,g))[:,1:2]
    Σ = round.(∇f * cov(prior) * transpose(∇f) + Q, digits=5) # Round to avoid issues with numerical precision
    MvNormal(f(vcat(mean(prior),Δt,g)),Σ)
end

# Update step
function kalman_update(prior, y, h, R)
    v = y - h(mean(prior))
    ∇h = jacobian(h,mean(prior))#[1:2] # returns a column vec
    S = ∇h' * cov(prior) * ∇h + R 	          # So transposes are reversed
    K = cov(prior) * ∇h * inv(S)
    Σ = round.(cov(prior) - K * S * transpose(K),digits=5) # Round to avoid numerical errors
    MvNormal(mean(prior) + K * v, Σ)
end
# Params
Δt = 0.01
g = 0.1
start = [0.,0.5]
T = 10000
σ = 0.1
# Process noise
Q = [0.00 0. ; 0. 0.01]
# Emission noise
R = 0.02
# Initial state
global prior = MvNormal([0.,0.], [0.1 0. ; 0. 0.1])
data = generate_data(start,T,g,Δt,σ)
plot([d[1] for d in data])

# Storage and simulation loop
x = Vector(undef,length(data))
for t ∈ 1:length(data)
    prediction = kalman_predict(prior, f, Q)
    global prior = kalman_update(prediction, data[t], h, R)
    x[t] = prior
end


# Plot 1st dimension
#plot(0:0.1:10,[d[1] for d in data])
#plot!(0:0.1:10,[d.μ[1] for d in x],ribbon=[d.Σ[1] for d in x])

# Plot 2nd dimension
plot([d[1] for d in data])
plot!(0:0.1:10,[d.μ[1] for d in x],ribbon=[d.Σ[1] for d in x])
#plot!(0:0.1:10,[d.μ[2] for d in x],ribbon=[d.Σ[4] for d in x])
