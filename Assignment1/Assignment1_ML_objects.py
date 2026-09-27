"""
    Author: Joe Ferguson, Maxwell Eichelberger
    Date: 2026-09-17
    All team members have contributed in equal measure to this effort.
"""

"""Library Imports - For Plotting and Randomization only - no ML libraries allowed"""

import random
import matplotlib.pyplot as plt

"""Function Definitions"""

### Separate data rows into feature names, feature values, and target values
def prepareRegressionData(data):
    # Identify the target column and every non-target feature column
    targetFeature = "quality"
    featureNames = []

    for name in data[0]:
        if name != targetFeature:
            featureNames.append(name)

    # Copy each row's feature values and target value into separate lists
    features = []
    target = []

    for rowIndex in range(len(data)):
        featureRow = []

        for featureIndex in range(len(featureNames)):
            featureName = featureNames[featureIndex]
            featureRow.append(data[rowIndex][featureName])

        features.append(featureRow)
        target.append(data[rowIndex][targetFeature])

    return featureNames, features, target

### Return predictions for each row in a feature set
def predict(features, weights):
    predictions = []

    # Calculate one prediction for each feature row
    for rowIndex in range(len(features)):
        row = features[rowIndex]
        prediction = 0

        # Add the contribution from each feature and weight
        for featureIndex in range(len(row)):
            prediction += row[featureIndex] * weights[featureIndex]

        predictions.append(prediction)

    return predictions

### Return the mean squared error for a set of predictions
def meanSquaredError(features, target, weights):
    # Calculate the predicted quality for every feature row
    predictions = predict(features, weights)
    squaredErrorTotal = 0

    # Add the squared error for each prediction
    for index in range(len(predictions)):
        error = predictions[index] - target[index]
        squaredErrorTotal += error ** 2

    return squaredErrorTotal / len(target)

### Return the gradient of the regularization penalty for a set of weights
def regGradient(weights, reg, regStrength):
    gradient = [0] * len(weights)

    # Add the regularization penalty for every non-bias weight
    if reg == "l2":
        # The gradient of the L2 penalty is 2 * alpha * w for every non-bias weight
        for index in range(1, len(weights)):
            gradient[index] = 2 * regStrength * weights[index]
    elif reg == "l1":
        # The gradient of the L1 penalty is 2 * alpha for every non-bias weight
        for index in range(1, len(weights)):
            if weights[index] > 0:
                gradient[index] = 2 * regStrength
            elif weights[index] < 0:
                gradient[index] = -2 * regStrength

    #if reg is None, the gradient is zero for every weight, which is already initialized
    return gradient

### Return the gradient of the mean squared error with optional regularization 
def getGradient(features, target, weights, reg, regStrength):
    # Initialize the gradient for every weight to zero
    predictions = predict(features, weights)
    sampleCount = len(features)
    gradient = [0] * len(weights)

    # Add the gradient of the mean squared error for every weight
    for rowIndex in range(len(features)):
        row = features[rowIndex]
        error = predictions[rowIndex] - target[rowIndex]

        for featureIndex in range(len(row)):
            gradient[featureIndex] += (
                2 * error * row[featureIndex] / sampleCount
            )

    # Add the gradient of the regularization penalty for every weight
    penaltyGradient = regGradient(
        weights, reg, regStrength
    )

    # Combine the data gradient and reg gradient
    for index in range(len(gradient)):
        gradient[index] += penaltyGradient[index]

    return gradient

### Train linear regression using batch gradient descent
def trainBatchGradientDescent(
    features,
    target,
    learningRate,
    epochs,
    reg,
    regStrength,
    seed=42,
    tolerance=1e-8,
):
    # Initialize weights list
    randomGenerator = random.Random(seed)
    weights = []

    # Create a small random starting weight for every feature
    for featureIndex in range(len(features[0])):
        weights.append(randomGenerator.uniform(-0.01, 0.01))

    # Update weights once for every epoch
    previousError = float("inf")
    for epoch in range(epochs):
        gradient = getGradient(features, target, weights, reg, regStrength)
        for weightIndex in range(len(weights)):
            weights[weightIndex] -= learningRate * gradient[weightIndex]

        # Stop early once the error stops improving meaningfully
        currentError = meanSquaredError(features, target, weights)
        if abs(previousError - currentError) < tolerance:
            break
        previousError = currentError
    return weights

### Train linear regression using mini-batch gradient descent
def trainMiniBatchGradientDescent(
    features,
    target,
    batchSize,
    learningRate,
    epochs,
    reg,
    regStrength,
    seed=42,
):
    # Initialize weights list
    randomGenerator = random.Random(seed)
    weights = []

    # Create a small random starting weight for every feature
    for featureIndex in range(len(features[0])):
        weights.append(randomGenerator.uniform(-0.01, 0.01))
    indices = list(range(len(features)))

    # Update weights once for every epoch using mini-batches
    for epoch in range(epochs):
        # Shuffle the row indices and process them in mini-batches
        randomGenerator.shuffle(indices)
        for start in range(0, len(indices), batchSize):
            batchIndices = indices[start : start + batchSize]
            batchFeatures = []
            batchTarget = []

            # Gather the feature rows and targets for this mini-batch
            for batchIndex in range(len(batchIndices)):
                rowIndex = batchIndices[batchIndex]
                batchFeatures.append(features[rowIndex])
                batchTarget.append(target[rowIndex])

            # Calculate one gradient after the entire mini-batch is collected
            gradient = getGradient(
                batchFeatures,
                batchTarget,
                weights,
                reg,
                regStrength,
            )

            # Update weights using the mini-batch gradient
            for weightIndex in range(len(weights)):
                weights[weightIndex] -= learningRate * gradient[weightIndex]

    return weights

### Select the best regularization strength for a given training and validation set
def selectRegStrength(
    features,
    target,
    validationFeatures,
    validationTarget,
    reg,
    trainFunction,
    batchSize,
    learningRate,
    epochs,
):
    # Define candidate regularization strengths to evaluate
    candidateStrengths = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    bestStrength = candidateStrengths[0]
    bestError = float("inf")

    # Evaluate each candidate strength and keep the one with the lowest validation error
    for strength in candidateStrengths:
        try:
            if trainFunction == trainMiniBatchGradientDescent:
                weights = trainFunction(
                    features,
                    target,
                    batchSize=batchSize,
                    reg=reg,
                    regStrength=strength,
                    learningRate=learningRate,
                    epochs=epochs,
                )
            else:
                weights = trainFunction(
                    features,
                    target,
                    reg=reg,
                    regStrength=strength,
                    learningRate=learningRate,
                    epochs=epochs,
                )

            # A strength this large can make gradient descent diverge; treat
            # a diverging fit as an infinitely bad candidate instead of crashing
            validationError = meanSquaredError(
                validationFeatures, validationTarget, weights
            )
        except OverflowError:
            validationError = float("inf")

        if validationError < bestError:
            bestError = validationError
            bestStrength = strength
            
    return bestStrength

### Remove one non-bias feature column from a data set
def removeFeature(features, featureIndex):
    reducedFeatures = []

    # Copy each row without the selected feature column
    for rowIndex in range(len(features)):
        # Create a new row without the selected feature
        reducedRow = []
        for columnIndex in range(len(features[rowIndex])):
            if columnIndex != featureIndex:
                reducedRow.append(features[rowIndex][columnIndex])

        reducedFeatures.append(reducedRow)

    return reducedFeatures

### Randomly split data into train, validation, and test sets
def splitData(features, target, trainRatio=0.70, validationRatio=0.15, seed=42):
    # Shuffle the row indices first
    totalRows = len(features)
    shuffledIndices = list(range(totalRows))
    random.Random(seed).shuffle(shuffledIndices)

    # Determine the ending index for each data set
    trainEnd = int(totalRows * trainRatio)
    validationEnd = trainEnd + int(totalRows * validationRatio)
    trainIndices = shuffledIndices[:trainEnd]
    validationIndices = shuffledIndices[trainEnd:validationEnd]
    testIndices = shuffledIndices[validationEnd:]

    # Create feature rows with a bias column for one set of indices
    def makeFeatureSet(indices):
        featureSet = []
        # Add a bias column and every feature value for each row in the selected indices
        for index in range(len(indices)):
            rowIndex = indices[index]
            featureRow = [1]
            # Add every feature value for this row to the feature row
            for featureIndex in range(len(features[rowIndex])):
                featureRow.append(features[rowIndex][featureIndex])

            featureSet.append(featureRow)

        return featureSet

    # Create the three feature sets
    trainFeatures = makeFeatureSet(trainIndices)
    validationFeatures = makeFeatureSet(validationIndices)
    testFeatures = makeFeatureSet(testIndices)

    # Create the target sets using the matching row indices
    trainTarget = []
    validationTarget = []
    testTarget = []

    for index in range(len(trainIndices)):
        trainTarget.append(target[trainIndices[index]])

    for index in range(len(validationIndices)):
        validationTarget.append(target[validationIndices[index]])

    for index in range(len(testIndices)):
        testTarget.append(target[testIndices[index]])

    # Print the number of rows in each set
    print("\nData split:")
    print("Training set rows: {}".format(len(trainFeatures)))
    print("Validation set rows: {}".format(len(validationFeatures)))
    print("Test set rows: {}".format(len(testFeatures)))

    return (
        trainFeatures,
        validationFeatures,
        testFeatures,
        trainTarget,
        validationTarget,
        testTarget,
    )

### Normalize features using the training-set mean and standard deviation
def normalizeFeatures(trainFeatures, validationFeatures, testFeatures):
    # Get the number of columns and prepare to save normalization values
    featureCount = len(trainFeatures[0])
    means = []
    standardDeviations = []

    # Calculate a mean and standard deviation for every non-bias column
    for columnIndex in range(1, featureCount):
        values = []

        for rowIndex in range(len(trainFeatures)):
            values.append(trainFeatures[rowIndex][columnIndex])

        average = sum(values) / len(values)
        varianceTotal = 0

        for valueIndex in range(len(values)):
            difference = values[valueIndex] - average
            varianceTotal += difference ** 2

        standardDeviation = (varianceTotal / len(values)) ** 0.5
        means.append(average)

        if standardDeviation == 0:
            standardDeviations.append(1.0)
        else:
            standardDeviations.append(standardDeviation)

    # Normalize every row in one data set with the training-set values
    def normalize(dataSet):
        normalizedData = []
        # Normalize every row in the data set 
        for rowIndex in range(len(dataSet)):
            normalizedRow = [1]

            # Normalize every non-bias column in the row
            for columnIndex in range(1, len(dataSet[rowIndex])):
                meanIndex = columnIndex - 1
                normalizedValue = (
                    dataSet[rowIndex][columnIndex] - means[meanIndex]
                ) / standardDeviations[meanIndex]
                normalizedRow.append(normalizedValue)

            normalizedData.append(normalizedRow)

        return normalizedData

    return normalize(trainFeatures), normalize(validationFeatures), normalize(testFeatures)

### Return the non-bias feature index with the smallest absolute weight
def smallestWeightFeature(weights):
    smallestIndex = 1
    for index in range(2, len(weights)):
        if abs(weights[index]) < abs(weights[smallestIndex]):
            smallestIndex = index

    return smallestIndex

### Return every non-bias feature index whose weight shrank essentially to zero
def nearZeroWeightFeatures(weights, threshold=0.01):
    nearZero = []
    for index in range(1, len(weights)):
        if abs(weights[index]) < threshold:
            nearZero.append(index)

    # Fall back to the single smallest weight if the threshold caught nothing
    if not nearZero:
        nearZero.append(smallestWeightFeature(weights))

    return nearZero

### Return the non-bias feature index with the largest absolute weight
def largestWeightFeature(weights):
    largestIndex = 1
    for index in range(2, len(weights)):
        if abs(weights[index]) > abs(weights[largestIndex]):
            largestIndex = index

    return largestIndex

### Plot regression lines before and after feature elimination
def plotRegressionComparison(
    beforeFeatures,
    beforeWeights,
    beforeFeatureNames,
    afterFeatures,
    afterWeights,
    afterFeatureNames,
    target,
    title,
):
    # Identify the feature represented in each comparison plot
    beforeIndex = largestWeightFeature(beforeWeights)
    afterIndex = largestWeightFeature(afterWeights)
    beforeName = beforeFeatureNames[beforeIndex - 1]
    afterName = afterFeatureNames[afterIndex - 1]

    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    featureSets = [beforeFeatures, afterFeatures]
    weightSets = [beforeWeights, afterWeights]
    featureIndices = [beforeIndex, afterIndex]
    featureNames = [beforeName, afterName]
    labels = ["Before feature elimination", "After feature elimination"]

    # Draw one plot before elimination and one after elimination
    for plotIndex in range(len(axes)):

        features = featureSets[plotIndex]
        weights = weightSets[plotIndex]
        featureIndex = featureIndices[plotIndex]
        xValues = []

        # Extract the feature values for this plot
        for rowIndex in range(len(features)):
            xValues.append(features[rowIndex][featureIndex])

        orderedX = sorted(xValues)
        lineValues = []

        # Extract the predicted values for the regression line at each x value
        for valueIndex in range(len(orderedX)):
            value = orderedX[valueIndex]
            lineValues.append(weights[0] + weights[featureIndex] * value)

        axes[plotIndex].scatter(xValues, target, alpha=0.2, s=8, label="Training data")
        axes[plotIndex].plot(orderedX, lineValues, color="darkorange", label="Regression line")
        axes[plotIndex].set_title("{}\nLargest |weight|: {}".format(
            labels[plotIndex], featureNames[plotIndex]
        ))
        axes[plotIndex].set_xlabel(featureNames[plotIndex])
        axes[plotIndex].set_ylabel("Quality")
        axes[plotIndex].legend()
        axes[plotIndex].grid(alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()

### Train the regression models and perform L1 and L2 feature elimination
def runRegressionAnalysis(
        features,
        target,
        featureNames,
        learningRate=0.01,
        epochs=200
    ):
    # Split the data once so every model uses the same rows
    Xtrain, Xvalid, Xtest, Ytrain, Yvalid, Ytest = splitData(
        features, target, seed=42
    )

    # Fit normalization on training data and reuse it for validation and test data
    Xtrain, Xvalid, Xtest = normalizeFeatures(Xtrain, Xvalid, Xtest)

    """ Step 1: Train and Test - Gradient Descent """

    print("\n=== Gradient Descent Results ===")

    # Select alpha on the validation set for each regularized method
    batchNoReg = selectRegStrength(
        Xtrain, Ytrain, Xvalid, Yvalid,
        reg=None, batchSize=None,
        learningRate=learningRate, epochs=epochs,
        trainFunction=trainBatchGradientDescent,
    )
    batchL1Strength = selectRegStrength(
        Xtrain, Ytrain, Xvalid, Yvalid,
        reg="l1", batchSize=None,
        learningRate=learningRate, epochs=epochs,
        trainFunction=trainBatchGradientDescent,
    )
    batchL2Strength = selectRegStrength(
        Xtrain, Ytrain, Xvalid, Yvalid,
        reg="l2", batchSize=None,
        learningRate=learningRate, epochs=epochs,
        trainFunction=trainBatchGradientDescent,
    )
    miniBatchNoReg = selectRegStrength(
        Xtrain, Ytrain, Xvalid, Yvalid,
        reg=None, batchSize=256,
        learningRate=learningRate, epochs=epochs,
        trainFunction=trainMiniBatchGradientDescent,
    )
    miniBatchL1Strength = selectRegStrength(
        Xtrain, Ytrain, Xvalid, Yvalid,
        reg="l1", batchSize=256,
        learningRate=learningRate, epochs=epochs,
        trainFunction=trainMiniBatchGradientDescent,
    )
    miniBatchL2Strength = selectRegStrength(
        Xtrain, Ytrain, Xvalid, Yvalid,
        reg="l2", batchSize=256,
        learningRate=learningRate, epochs=epochs,
        trainFunction=trainMiniBatchGradientDescent,
    )

    # Train the batch and mini-batch models without regularization
    batchWeights = trainBatchGradientDescent(
        Xtrain, Ytrain,
        reg=None, regStrength=batchNoReg,
        learningRate=learningRate, epochs=epochs,
    )

    miniBatchWeights = trainMiniBatchGradientDescent(
        Xtrain, Ytrain,
        batchSize=256, reg=None, regStrength=miniBatchNoReg,
        learningRate=learningRate, epochs=epochs,
    )

    # Train all six models on the full feature set at their selected alphas
    sixModels = [
        ("Batch, no regularization",   batchWeights),
        ("Batch, L2 (ridge)",          trainBatchGradientDescent(
            Xtrain, Ytrain, reg="l2", regStrength=batchL2Strength,
            learningRate=learningRate, epochs=epochs)),
        ("Batch, L1 (lasso)",          trainBatchGradientDescent(
            Xtrain, Ytrain, reg="l1", regStrength=batchL1Strength,
            learningRate=learningRate, epochs=epochs)),
        ("Mini-batch, no regularization", miniBatchWeights),
        ("Mini-batch, L2 (ridge)",     trainMiniBatchGradientDescent(
            Xtrain, Ytrain, batchSize=256, reg="l2", regStrength=miniBatchL2Strength,
            learningRate=learningRate, epochs=epochs)),
        ("Mini-batch, L1 (lasso)",     trainMiniBatchGradientDescent(
            Xtrain, Ytrain, batchSize=256, reg="l1", regStrength=miniBatchL1Strength,
            learningRate=learningRate, epochs=epochs)),
    ]

    print("\n--- All Six Models (full feature set) ---")
    for modelName, modelWeights in sixModels:
        print("{:32s} train {:.4f}  valid {:.4f}  test {:.4f}".format(
            modelName,
            meanSquaredError(Xtrain, Ytrain, modelWeights),
            meanSquaredError(Xvalid, Yvalid, modelWeights),
            meanSquaredError(Xtest, Ytest, modelWeights),
        ))

    # Print the training, validation, and test MSE for each model
    print("\n--- Batch Gradient Descent ---")
    print("Train MSE:      {:.4f}".format(meanSquaredError(Xtrain, Ytrain, batchWeights)))
    print("Validation MSE: {:.4f}".format(meanSquaredError(Xvalid, Yvalid, batchWeights)))
    print("Test MSE:       {:.4f}".format(meanSquaredError(Xtest, Ytest, batchWeights)))

    print("\n--- Mini-Batch Gradient Descent ---")
    print("Train MSE:      {:.4f}".format(meanSquaredError(Xtrain, Ytrain, miniBatchWeights)))
    print("Validation MSE: {:.4f}".format(meanSquaredError(Xvalid, Yvalid, miniBatchWeights)))
    print("Test MSE:       {:.4f}".format(meanSquaredError(Xtest, Ytest, miniBatchWeights)))

    """ Step 2: L2 reg and Feature Elimination """

    # Train L2 regularized models and eliminate the feature with the smallest absolute weight
    l2Weights = trainBatchGradientDescent(
        Xtrain, Ytrain,
        reg="l2", regStrength=batchL2Strength,
        learningRate=learningRate, epochs=epochs,
    )

    l2FeatureIndex = smallestWeightFeature(l2Weights)
    l2FeatureName = featureNames[l2FeatureIndex - 1]
    XtrainL2 = removeFeature(Xtrain, l2FeatureIndex)
    XvalidL2 = removeFeature(Xvalid, l2FeatureIndex)
    XtestL2 = removeFeature(Xtest, l2FeatureIndex)
    l2ReducedWeights = trainBatchGradientDescent(
        XtrainL2, Ytrain,
        reg="l2", regStrength=batchL2Strength,
        learningRate=learningRate, epochs=epochs,
    )
    l2FeatureNames = []

    for index in range(len(featureNames)):
        if index + 1 != l2FeatureIndex:
            l2FeatureNames.append(featureNames[index])

    plotRegressionComparison(
        Xtrain, batchWeights, featureNames,
        XtrainL2, l2ReducedWeights, l2FeatureNames,
        Ytrain,
        "Batch Gradient Descent: L2 Feature Elimination",
    )

    print("\n--- Batch Gradient Descent with L2 Regularization ---")
    print("Selected alpha: {}".format(batchL2Strength))
    print("Eliminated feature: {}".format(l2FeatureName))
    print("Train MSE after retraining:      {:.4f}".format(
        meanSquaredError(XtrainL2, Ytrain, l2ReducedWeights)
    ))
    print("Validation MSE after retraining: {:.4f}".format(
        meanSquaredError(XvalidL2, Yvalid, l2ReducedWeights)
    ))
    print("Test MSE after retraining:       {:.4f}".format(
        meanSquaredError(XtestL2, Ytest, l2ReducedWeights)
    ))

    # Train mini-batch L2 regularized models and eliminate the feature with the smallest absolute weight
    miniL2Weights = trainMiniBatchGradientDescent(
        Xtrain, Ytrain,
        batchSize=256, reg="l2", regStrength=miniBatchL2Strength,
        learningRate=learningRate, epochs=epochs,
    )

    miniL2FeatureIndex = smallestWeightFeature(miniL2Weights)
    miniL2FeatureName = featureNames[miniL2FeatureIndex - 1]
    XtrainMiniL2 = removeFeature(Xtrain, miniL2FeatureIndex)
    XvalidMiniL2 = removeFeature(Xvalid, miniL2FeatureIndex)
    XtestMiniL2 = removeFeature(Xtest, miniL2FeatureIndex)
    miniL2ReducedWeights = trainMiniBatchGradientDescent(
        XtrainMiniL2, Ytrain,
        batchSize=256, reg="l2", regStrength=miniBatchL2Strength,
        learningRate=learningRate, epochs=epochs,
    )
    miniL2FeatureNames = []

    for index in range(len(featureNames)):
        if index + 1 != miniL2FeatureIndex:
            miniL2FeatureNames.append(featureNames[index])

    plotRegressionComparison(
        Xtrain, miniBatchWeights, featureNames,
        XtrainMiniL2, miniL2ReducedWeights, miniL2FeatureNames,
        Ytrain,
        "Mini-Batch Gradient Descent: L2 Feature Elimination",
    )

    print("\n--- Mini-Batch Gradient Descent with L2 Regularization ---")
    print("Selected alpha: {}".format(miniBatchL2Strength))
    print("Eliminated feature: {}".format(miniL2FeatureName))
    print("Train MSE after retraining:      {:.4f}".format(
        meanSquaredError(XtrainMiniL2, Ytrain, miniL2ReducedWeights)
    ))
    print("Validation MSE after retraining: {:.4f}".format(
        meanSquaredError(XvalidMiniL2, Yvalid, miniL2ReducedWeights)
    ))
    print("Test MSE after retraining:       {:.4f}".format(
        meanSquaredError(XtestMiniL2, Ytest, miniL2ReducedWeights)
    ))
    """ Step 3: L1 reg and Feature Elimination """

    # Train L1 regularized models and eliminate the feature with the smallest absolute weight
    l1Weights = trainBatchGradientDescent(
        Xtrain, Ytrain,
        reg="l1", regStrength=batchL1Strength,
        learningRate=learningRate, epochs=epochs,
    )

    l1FeatureIndex = smallestWeightFeature(l1Weights)
    l1FeatureName = featureNames[l1FeatureIndex - 1]
    XtrainL1 = removeFeature(Xtrain, l1FeatureIndex)
    XvalidL1 = removeFeature(Xvalid, l1FeatureIndex)
    XtestL1 = removeFeature(Xtest, l1FeatureIndex)
    l1ReducedWeights = trainBatchGradientDescent(
        XtrainL1, Ytrain,
        reg="l1", regStrength=batchL1Strength,
        learningRate=learningRate, epochs=epochs,
    )
    l1FeatureNames = []

    for index in range(len(featureNames)):
        if index + 1 != l1FeatureIndex:
            l1FeatureNames.append(featureNames[index])

    plotRegressionComparison(
        Xtrain, batchWeights, featureNames,
        XtrainL1, l1ReducedWeights, l1FeatureNames,
        Ytrain,
        "Batch Gradient Descent: L1 Feature Elimination",
    )

    print("\n--- Batch Gradient Descent with L1 Regularization ---")
    print("Selected alpha: {}".format(batchL1Strength))
    print("Eliminated feature: {}".format(l1FeatureName))
    print("Train MSE after retraining:      {:.4f}".format(
        meanSquaredError(XtrainL1, Ytrain, l1ReducedWeights)
    ))
    print("Validation MSE after retraining: {:.4f}".format(
        meanSquaredError(XvalidL1, Yvalid, l1ReducedWeights)
    ))
    print("Test MSE after retraining:       {:.4f}".format(
        meanSquaredError(XtestL1, Ytest, l1ReducedWeights)
    ))

    # Train mini-batch L1 regularized models and eliminate the feature with the smallest absolute weight
    miniL1Weights = trainMiniBatchGradientDescent(
        Xtrain, Ytrain,
        batchSize=256, reg="l1", regStrength=miniBatchL1Strength,
        learningRate=learningRate, epochs=epochs,
    )

    miniL1FeatureIndex = smallestWeightFeature(miniL1Weights)
    miniL1FeatureName = featureNames[miniL1FeatureIndex - 1]
    XtrainMiniL1 = removeFeature(Xtrain, miniL1FeatureIndex)
    XvalidMiniL1 = removeFeature(Xvalid, miniL1FeatureIndex)
    XtestMiniL1 = removeFeature(Xtest, miniL1FeatureIndex)
    miniL1ReducedWeights = trainMiniBatchGradientDescent(
        XtrainMiniL1, Ytrain,
        batchSize=256, reg="l1", regStrength=miniBatchL1Strength,
        learningRate=learningRate, epochs=epochs,
    )
    miniL1FeatureNames = []

    for index in range(len(featureNames)):
        if index + 1 != miniL1FeatureIndex:
            miniL1FeatureNames.append(featureNames[index])

    plotRegressionComparison(
        Xtrain, miniBatchWeights, featureNames,
        XtrainMiniL1, miniL1ReducedWeights, miniL1FeatureNames,
        Ytrain,
        "Mini-Batch Gradient Descent: L1 Feature Elimination",
    )

    print("\n--- Mini-Batch Gradient Descent with L1 Regularization ---")
    print("Selected alpha: {}".format(miniBatchL1Strength))
    print("Eliminated feature: {}".format(miniL1FeatureName))
    print("Train MSE after retraining:      {:.4f}".format(
        meanSquaredError(XtrainMiniL1, Ytrain, miniL1ReducedWeights)
    ))
    print("Validation MSE after retraining: {:.4f}".format(
        meanSquaredError(XvalidMiniL1, Yvalid, miniL1ReducedWeights)
    ))
    print("Test MSE after retraining:       {:.4f}".format(
        meanSquaredError(XtestMiniL1, Ytest, miniL1ReducedWeights)
    ))

    # Display the L1 and L2 comparison plots
    plt.show()