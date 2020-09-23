# Extended KF II, non additive noise terms. Needs proper transition/emission functions to not degenerate into completely deterministic system
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
	states[t+1] = f(vcat(states[t],0.)) # No noise since we just want observations
	obs[t] = h(states[t])
    end
    noise = [(rand(size(o)[1]) .- 0.5) .* σ for o in obs]
    return obs .+ noise,states
end

# Transition function. Does not depend on noise, so technically wrong
function f(x)
    #g = 0.1
    [x[1] + x[2] * Δt, x[2] - 1.0 * sin(x[1]) * Δt]
end

# Emission function. Does not depend on noise, so technically wrong
function h(x)
    x[1:2]
end


# Prediction step
function kalman_predict(prior, f, Q)
    idx = size(prior)[1]# Get dim of state
    ∇f = jacobian(f,vcat(mean(prior),zeros(idx))) # Jacobian of f. Augment with extra zeros
    ∇f_x = ∇f[:,1:idx]	# Portion that pertains to x
    ∇f_q = ∇f[:,idx+1:end]# Portion that pertains to q
    Σ = round.(∇f_x * cov(prior) * ∇f_x' + ∇f_q * Q * ∇f_q', digits=5) # Round to avoid issues with numerical precision
    MvNormal(f(mean(prior)),Σ + I*eps()) # Add epsilon to avoid zero matrices
end

# Update step
function kalman_update(prior, y, h, R)
    idx = size(prior)[1]	 # Get dim of state
    v = y - h(vcat(mean(prior),zeros(idx)))

    ∇h = jacobian(h,vcat(mean(prior),zeros(idx)))
    ∇h_x = ∇h[:,1:idx]	# Portion that pertains to x
    ∇h_q = ∇h[:,idx+1:end] # Portion that pertains to q

    S = ∇h_x * cov(prior) * ∇h_x' + ∇h_q * R * ∇h_q'
    K = cov(prior) * ∇h_x' * inv(S)
    Σ = round.(cov(prior) - K * S * transpose(K),digits=5) # Round to avoid numerical errors
    MvNormal(mean(prior) + K * v, Σ + I*eps()) # Epsilon to avoid getting a zero matrix when mapping is deterministic
end

# Params
Δt = 0.01
start = [1.,1.]
T = 1000
σ = [1.0,1.0]
# Process noise. Currently does nothing
Q = [.1 0. ; 0. .1]
# Emission noise. Currently does nothing
R = [σ[1] 0.0 ; 0.0 σ[2]]

# Initial state
global prior = MvNormal(start, [1.0 0. ; 0. 1.0])
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
