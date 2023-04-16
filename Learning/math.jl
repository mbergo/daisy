using CSV
using DataFrames
using GLM

function main()
    # Load the dataset
    data = CSV.read("math_data.csv", DataFrame)

    # Fit a linear regression model
    model = lm(@formula(Score ~ HoursStudied + ProblemSetsCompleted), data)

    # Display the model
    println("Linear Regression Model:")
    println(model)

    # Make predictions
    new_data = DataFrame(HoursStudied = [10, 15], ProblemSetsCompleted = [5, 7])
    predictions = predict(model, new_data)
    println("Predictions:")
    println(predictions)
end

main()
