import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


# Loading CSV file
df = pd.read_csv('D:\obesity_prediction\obesity_prediction\obesity_data.csv') 

# Checking the first few rows
print(df.head())
print(df.loc[345])
print(df.loc[346])
print(df.loc[344])
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

#Groups by age : 0-9,10-19,....90-99
df['AgeGroup'] = pd.cut(df['Age'], bins=range(0, 101, 10), right=False)

# Group by AgeGroup and calculate mean Weight
age_weight_group = df.groupby('AgeGroup')['Weight'].mean().reset_index()
print("🔹 Mean Weight by Age Group:")
print(age_weight_group)

# Get minimum and maximum values
min_values = df.min(numeric_only=True)
max_values = df.max(numeric_only=True)

# Combine into a single DataFrame
min_max_df = pd.DataFrame({'Minimum': min_values, 'Maximum': max_values})

# Display result
print("🔹 Minimum and Maximum Values for Each Column:")
print(min_max_df)

# Select only numeric columns
numeric_cols = ['Age', 'Height', 'Weight', 'BMI', 'PhysicalActivityLevel']

# Calculate Z-scores
z_scores = zscore(df[numeric_cols])

# Create a DataFrame of Z-scores
z_df = pd.DataFrame(z_scores, columns=numeric_cols)

# Find rows where any z-score is above 3 or below -3
outlier_rows = (z_df > 3) | (z_df < -3)
df_outliers = df[outlier_rows.any(axis=1)]

# Show detected outliers
print(f"🔍 Found {len(df_outliers)} outliers using Z-score method:")
print(df_outliers)