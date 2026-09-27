"""
    Author: Joe Ferguson, Maxwell Eichelberger
    Date: 2026-09-17
    All team members have contributed in equal measure to this effort.
"""

"""Library Imports - For Plotting and Randomization only - no ML libraries allowed"""

import os
import matplotlib.pyplot as plt
from Assignment1_ML_objects import prepareRegressionData, runRegressionAnalysis

"""Function Definitions"""

### Read the CSV file and return a list of the rows, where each row is represented as a dictionary
def readData(filePath, isRed):
    # Open the file
    with open(filePath, "r") as file:
        lines = []
        for line in file:
            strippedLine = line.strip()
            if strippedLine:
                lines.append(strippedLine)

    # Extract the header row
    headers = []
    rawHeaders = lines[0].split(";")
    for headerIndex in range(len(rawHeaders)):
        headers.append(rawHeaders[headerIndex].strip('"'))

    # Extract the rest of the rows
    rows = []
    for line in lines[1:]:
        values = line.split(";")
        row = {}
        for columnIndex in range(len(headers) - 1):
            header = headers[columnIndex]
            value = values[columnIndex]
            row[header] = float(value) if value.strip() else None

        # Grab the y/target value at the end
        row["quality"] = int(values[-1])

        # Add the isRed feature to the row
        row["isRed"] = isRed
        rows.append(row)
    return rows

### Run exploratory data analysis
def exploratoryDataAnalysis(rows):
    # Grab feature names
    featureNames = []
    for name in rows[0]:
        if name != "isRed":
            featureNames.append(name)

    # Print the number of missing values for each feature
    def missingValues(rows):
        featureNames = []
        for name in rows[0]:
            if name != "isRed":
                featureNames.append(name)
        print("\nMissing values\n")
        for name in featureNames:
            missingCount = 0
            for rowIndex in range(len(rows)):
                if rows[rowIndex][name] is None:
                    missingCount += 1
            print("  {}: {}".format(name, missingCount))

    # Extract numeric values for a given feature from the selected rows
    def numericValues(name, selectedRows):
        values = []
        for rowIndex in range(len(selectedRows)):
            values.append(selectedRows[rowIndex][name])
        return values

    # Compute the mean
    def mean(values):
        return sum(values) / len(values)

    # Compute the percentile
    def percentile(values, percent):
        ordered = sorted(values)
        position = (len(ordered) - 1) * percent / 100
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction

    # Compute the standard deviation
    def standardDeviation(values):
        average = mean(values)
        varianceTotal = 0
        for valueIndex in range(len(values)):
            varianceTotal += (values[valueIndex] - average) ** 2
        variance = varianceTotal / len(values)
        return variance ** 0.5

    # Create one histogram subplot for each feature
    def plotFeatureDistributions():
        # Determine the number of rows and columns for the subplots
        columnCount = 3
        rowCount = (len(featureNames) + columnCount - 1) // columnCount
        figure, axes = plt.subplots(rowCount, columnCount, figsize=(15, 12))
        axes = axes.ravel()

        # Plot a histogram for each feature in its own subplot.
        for index in range(len(featureNames)):
            name = featureNames[index]
            values = numericValues(name, rows)
            axes[index].hist(values, bins=5, edgecolor="black")
            axes[index].set_title(name)
            axes[index].set_ylabel("Count")
            axes[index].grid(axis="y", alpha=0.3)

        # Hide unused subplot spaces when the feature count is not a multiple of three, because we have three columns
        for axis in axes[len(featureNames):]:
            axis.set_visible(False)

        # Add a shared title and adjust spacing so labels do not overlap.
        figure.suptitle("Feature Distributions", fontsize=16)
        figure.tight_layout()

    # Plot a bar chart for the quality distribution 
    def plotQualityDistribution():
        # Count the number of rows at each quality level
        qualityValues = numericValues("quality", rows)
        qualityCounts = {}
        for value in qualityValues:
            qualityCounts[value] = qualityCounts.get(value, 0) + 1

        # Sort and get counts of each quality level
        qualities = sorted(qualityCounts)
        counts = []
        qualityLabels = []
        for qualityIndex in range(len(qualities)):
            quality = qualities[qualityIndex]
            counts.append(qualityCounts[quality])
            qualityLabels.append(str(quality))

        # Create a bar chart showing the number of rows at each quality level.
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(qualityLabels, counts)
        axis.set_title("Wine Quality Distribution")
        axis.set_xlabel("Quality")
        axis.set_ylabel("Count")
        axis.grid(axis="y", alpha=0.3)
        figure.tight_layout()

    # Plot summary statistics for each feature 
    def plotSummaryStatistics():
        summary = []

        # Determine the quartiles, min, max, ave, and store them
        for name in featureNames:
            values = sorted(numericValues(name, rows))
            minimum = values[0]
            maximum = values[-1]
            firstQuartile = percentile(values, 25)
            median = percentile(values, 50)
            thirdQuartile = percentile(values, 75)
            average = mean(values)
            summary.append((name, minimum, firstQuartile, median, thirdQuartile, maximum, average))

        positions = list(range(len(summary)))

        # Draw each feature's range, interquartile range, median, and mean.
        figure, axis = plt.subplots(figsize=(12, 8))
        for position in range(len(positions)):
            summaryRow = summary[position]
            minimum = summaryRow[1]
            firstQuartile = summaryRow[2]
            median = summaryRow[3]
            thirdQuartile = summaryRow[4]
            maximum = summaryRow[5]
            average = summaryRow[6]
            axis.plot([minimum, maximum], [position, position], color="lightgray", linewidth=2)
            axis.plot([firstQuartile, thirdQuartile], [position, position], color="steelblue", linewidth=8)
            axis.scatter(median, position, color="black", label="Median" if position == 0 else None, zorder=3)
            axis.scatter(average, position, color="darkorange", label="Mean" if position == 0 else None, zorder=3)

        # Add labels, title, legend, and grid
        axis.set_yticks(positions)
        featureLabels = []
        for summaryIndex in range(len(summary)):
            featureLabels.append(summary[summaryIndex][0])
        axis.set_yticklabels(featureLabels)
        axis.set_xlabel("Feature value")
        axis.set_title("Feature Summary Statistics")
        axis.legend()
        axis.grid(axis="x", alpha=0.3)
        figure.tight_layout()

    # Print missing values
    missingValues(rows)

    # Print exploratory data analysis results
    print("\nExploratory data analysis\n")
    print("Rows: {}".format(len(rows)))
    redWineCount = 0
    whiteWineCount = 0
    for rowIndex in range(len(rows)):
        if rows[rowIndex]["isRed"] == 1:
            redWineCount += 1
        else:
            whiteWineCount += 1
    print("Red wine rows: {}".format(redWineCount))
    print("White wine rows: {}".format(whiteWineCount))

    plotFeatureDistributions()
    plotQualityDistribution()
    plotSummaryStatistics()

    # Display all three figures
    plt.show()

""" Step 0: Read and Explore the Data """

### Read the data - Need to remove hardcoded file paths
scriptDirectory = os.path.dirname(os.path.abspath(__file__))

redWine = readData(os.path.join(scriptDirectory, "winequality-red.csv"), isRed=1)
whiteWine = readData(os.path.join(scriptDirectory, "winequality-white.csv"), isRed=0)
data = redWine + whiteWine

# Run the exploratory data analysis
print("Combined rows: {}".format(len(data)))
exploratoryDataAnalysis(data)

# Separate the data, then run the regression workflow from the ML module
featureNames, features, target = prepareRegressionData(data)
runRegressionAnalysis(features, target, featureNames, learningRate=0.01, epochs=300)
