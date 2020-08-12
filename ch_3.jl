using Pkg;Pkg.activate(".");Pkg.instantiate()
using Distributions, LinearAlgebra, Plots, OhMyREPL
# This fixes errors with sending lines to the REPL
enable_autocomplete_brackets(false)

function generate_sin_data(minimum,maximum,freq,σ)
    [sin(x) + rand(Normal(0,σ)) for x in minimum:freq:maximum]
end

data = generate_sin_data(0,10,0.1,0.1)

# Transition matrix and noise
A = [1. 1. ; 0. 1.]
q = MvNormal([0., 0.],[1/10^2 0. ; 0. 1^2])
# Emission matrix and noise
F = [ 1.; 0.]
r = Normal(0.,1/1)

# Initialize variables
x_0 = rand(MvNormal([0.,0.], [1. 0. ; 0. 1.]))
x_vec = zeros(2,101)
x_vec[:,1] = x_0
y_vec = zeros(1,101)
y_vec[1] = dot(F, x_vec[:,1]) + rand(r)

# Generate data from model
for t in 2:101
    x_vec[:,t] = A*x_vec[:,t-1] + rand(q)
    y_vec[t] = dot(F, x_vec[:,t]) + rand(r)
end

# plots
plot(1:101,y_vec[:],label="Emissions")
plot!(1:101,x_vec[1,:],label="Position")
plot!(1:101,x_vec[2,:],label="Velocity")


