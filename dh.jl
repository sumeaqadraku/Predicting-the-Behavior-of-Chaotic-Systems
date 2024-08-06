# dh.jl

using DifferentialEquations
using Plots
using DataFrames
using GLM

# Define the Lorenz system
function lorenz!(du, u, p, t)
    du[1] = 10.0 * (u[2] - u[1])
    du[2] = u[1] * (28.0 - u[3]) - u[2]
    du[3] = u[1] * u[2] - 8.0 / 3.0 * u[3]
end

# Initial conditions and time span
u0 = [1.0, 0.0, 0.0]
tspan = (0.0, 30.0)
prob = ODEProblem(lorenz!, u0, tspan)
sol = solve(prob)

# Prepare data for machine learning
df = DataFrame(x1=sol[1], x2=sol[2], x3=sol[3], t=sol.t)

# Split data into training and testing sets
train_size = Int(0.8 * length(df.t))
train_df = df[1:train_size, :]
test_df = df[(train_size + 1):end, :]

# Fit a linear model
model = lm(@formula(x1 ~ x2 + x3), train_df)

# Predict
predictions = predict(model, test_df)

# Plot actual vs predicted
plot(test_df.t, test_df.x1, label="Actual", xlabel="Time", ylabel="x1", title="Lorenz System Prediction")
plot!(test_df.t, predictions, label="Predicted", linestyle=:dash)
