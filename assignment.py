# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris

# Set Seaborn style
sns.set(style="whitegrid")

# Task 1: Load and Clean the Dataset

try:
    # Load the Iris dataset from sklearn
    iris = load_iris()
    
    # Convert to DataFrame
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    
    # Add the target (species) as a column
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    # Display the first few rows
    print("First 5 rows of the dataset:")
    print(df.head())

    # Check data types and missing values
    print("\nDataset Info:")
    print(df.info())

    print("\nMissing Values:")
    print(df.isnull().sum())

    # No missing values in this dataset, but if there were:
    # df = df.dropna()  # or use df.fillna(value)

except FileNotFoundError:
    print("Error: File not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Task 2: Basic Data Analysis

# Compute basic statistics
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute mean of numerical features
grouped = df.groupby('species').mean()
print("\nMean values grouped by species:")
print(grouped)

# Interesting Finding (printed)
print("\nInteresting Findings:")
print("- Setosa species has significantly smaller petal lengths and widths compared to Versicolor and Virginica.")
print("- Virginica tends to have the largest sepal and petal dimensions on average.")

# Task 3: Data Visualization

# Line Chart: Simulating a time series using index
plt.figure(figsize=(10, 5))
plt.plot(df.index, df['sepal length (cm)'], label='Sepal Length', color='blue')
plt.plot(df.index, df['petal length (cm)'], label='Petal Length', color='green')
plt.title('Simulated Time Series of Sepal and Petal Length')
plt.xlabel('Index')
plt.ylabel('Length (cm)')
plt.legend()
plt.tight_layout()
plt.show()

# Bar Chart: Average petal length per species
plt.figure(figsize=(8, 6))
sns.barplot(data=df, x='species', y='petal length (cm)', palette='pastel')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.show()

# Histogram: Distribution of Sepal Width
plt.figure(figsize=(8, 6))
sns.histplot(df['sepal width (cm)'], bins=15, kde=True, color='coral')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Scatter Plot: Sepal Length vs Petal Length
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species', palette='deep')
plt.title('Sepal Length vs Petal Length by Species')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.tight_layout()
plt.show()
