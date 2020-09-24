# Extended Kalman filter III. Second order Taylor expansion
using Pkg;Pkg.activate(".");Pkg.instantiate()
using OhMyREPL, Plots, Distributions, ForwardDiff, LinearAlgebra
enable_autocomplete_brackets(false)

# shorthand
jacobian = ForwardDiff.jacobian
gradient = ForwardDiff.gradient
function hessian(f,x)
    n = length(x)
    hess = ForwardDiff.jacobian( x -> ForwardDiff.jacobian(f,x),x)
    reshape(hess, n, n, n)
end


# Generate data from model
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
    x
end



# Prediction step
function kalman_predict(prior, f, Q)
    idx = size(prior)[1]# Get dim of state
    e = I(idx) # stack of e vectors from Sarkka.

    ∇f = jacobian(f,mean(prior))
    ∇∇f = hessian(f,mean(prior))

    μ = f(mean(prior)) + 0.5 * [tr(∇∇f[:,:,i] * cov(prior)) for i in 1:idx]

    # List comprehension in the middle computes array. Round to avoid numerical errors
    Σ = round.(∇f * cov(prior) * ∇f' + 0.5 * [e[i] * e'[j]* tr(∇∇f[:,:,i] * cov(prior) * ∇∇f[:,:,j] * cov(prior)) for i in 1:idx, j in 1:idx]+ Q, digits=5)

    MvNormal(μ,Σ + I*eps()) # Add epsilon to avoid zero diagonals
end

# Update step
function kalman_update(prior, y, h, R)
    idx = size(prior)[1]	 # Get dim of state
    e = I(idx) # stack of e vectors from Sarkka.

    ∇h = jacobian(h,mean(prior))
    ∇∇h = hessian(h,mean(prior))

    v = y - h(mean(prior)) - 0.5 * [tr(∇∇h[:,:,i] * cov(prior)) for i in 1:idx]

    # List comprehension with e to handle the sum
    S = ∇h * cov(prior) * ∇h'+ 0.5* [e[i] * e'[j]* tr(∇∇h[:,:,i] * cov(prior) * ∇∇h[:,:,j] * cov(prior)) for i in 1:idx, j in 1:idx] + R

    K = cov(prior) * ∇h' * inv(S)
    Σ = round.(cov(prior) - K * S * K',digits=5) # Round to avoid numerical errors

    MvNormal(mean(prior) + K * v, Σ + I*eps()) # Epsilon to avoid getting a zero matrix when mapping is deterministic
end

# Params
Δt = 0.01
start = [1.,1.]
T = 1000
σ = [1.0,1.0]
# Process noise.
Q = [.1 0. ; 0. .1]
# Emission noise.
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
