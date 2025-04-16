import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Loading CSV file
df = pd.read_csv('obesity_data.csv') 

# Checking the first few rows
print(df.head())

# Checking structure
print("\n🔹 Data Types & Nulls:")
print(df.info())

# Checking missing values
print("\n🔹 Missing Values:")
print(df.isnull().sum())



# Summary statistics
print("\n🔹 Summary Statistics:")
print(df.describe())

# Calculating mean, median, and standard deviation
metrics = df[['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']].agg(['mean', 'median', 'std'])

# Displaying the results
print("🔹 Basic Metrics (Mean, Median, Standard Deviation):")
print(metrics)

