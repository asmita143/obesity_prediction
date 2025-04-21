import pandas as pd
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Loading CSV file
df = pd.read_csv('D:\\obesity_prediction\\obesity_prediction\\obesity_data_detailed.csv') 

# Checking the first few rows
print("* First 5 rows of the dataset:")
print(df.head())

# Checking structure
print("\n * Info about the dataset:")
print(df.info())

# Checking if 'Weight' and 'Height' columns exist
if 'Weight' in df.columns and 'Height' in df.columns:
    # Calculate BMI and add it as a new column
    df['BMI'] = df['Weight'] / (df['Height'] ** 2)
    print("\n * BMI column added successfully!")
else:
    print("\n * 'Weight' or 'Height' columns are missing in the dataset. Please ensure these columns exist.")

# Summary statistics
print("\n * Summary Statistics:")
print(df.describe())

# Checking missing values
print("\n * Missing Values in each column:")
print(df.isnull().sum())

# Selecting only numerical columns
numeric_cols = df.select_dtypes(include=['float64', 'int64'])

# Calculating and printing basic statistics
print("📈 Mean values:\n", numeric_cols.mean())
print("\n📊 Median values:\n", numeric_cols.median())
print("\n📉 Standard Deviation:\n", numeric_cols.std())

# Calculate z-scores for numerical columns
z_scores = zscore(numeric_cols)

# Convert to DataFrame for easier filtering
z_df = pd.DataFrame(z_scores, columns=numeric_cols.columns)

# Find rows where any z-score is above 3 or below -3
outliers = df[(abs(z_df) > 3).any(axis=1)]

print(f"🚨 Found {len(outliers)} outliers using Z-score method:\n")
print(outliers)

# Get minimum and maximum values
min_values = df.min(numeric_only=True)
max_values = df.max(numeric_only=True)

# Combine into a single DataFrame
min_max_df = pd.DataFrame({'Minimum': min_values, 'Maximum': max_values})

# Display result
print("\n * Minimum and Maximum Values for Each Column:")
print(min_max_df)

print(f"📐 Dataset shape (rows, columns): {df.shape}")
 
# AGE GROUP VS OBESITY LEVEL ANALYSIS
# Creating age groups
bins = [0, 20, 40, 60, 80] 
labels = ['0-20', '21-40','41-60','61-80']  
df['AgeGroup'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False) 

# Grouping by AgeGroup and Obesity Category, then counting occurrences
age_obesity_group = df.groupby(['AgeGroup', 'NObeyesdad'], observed=False).size().reset_index(name='Count')

# Plotting a bar chart for the distribution of obesity levels
plt.figure(figsize=(8, 6))
sns.barplot(x='AgeGroup', y='Count', hue='NObeyesdad', data=age_obesity_group)
plt.title('Obesity Level Distribution Across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Number of Individuals')
plt.legend(title='Obesity Category', loc='upper left')
plt.show()

# GENDER BASED ANALYSIS OF OBESITY
# Grouping data by Gender and Obesity Category
gender_obesity_group = df.groupby(['Gender', 'NObeyesdad'], observed=False).size().reset_index(name='Count')
# Plotting the Gender-based Obesity Distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Gender', hue='NObeyesdad', data=df)
plt.title('Gender-Based Obesity Distribution')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Obesity Category', loc='upper right')
plt.show()

# LIFESTYLE FACTORS VS OBESITY RISK

# Analyzing Physical Activity Level (FAF) vs Obesity Category
plt.figure(figsize=(10, 6))
sns.boxplot(x='NObeyesdad', y='FAF', data=df)
plt.title('Distribution of Physical Activity Level by Obesity Category')
plt.xlabel('Obesity Category')
plt.ylabel('Physical Activity Level (FAF)')
plt.xticks(rotation=45)
plt.show()

# Analyzing Family History of Obesity vs Obesity Category
plt.figure(figsize=(8, 6))
sns.countplot(x='family_history_with_overweight', hue='NObeyesdad', data=df)
plt.title('Family History of Obesity vs Obesity Risk')
plt.xlabel('Family History of Obesity (Yes/No)')
plt.ylabel('Count')
plt.legend(title='Obesity Category', loc='upper left')
plt.show()

# Stacked bar plot for FAVC vs Obesity Level
pd.crosstab(df['FAVC'], df['NObeyesdad']).plot(
    kind='bar', 
    stacked=True, 
    colormap='coolwarm', 
    figsize=(10, 6)
)
plt.title('Fast Food Consumption vs. Obesity Levels')
plt.xlabel('Consumes Fast Food (FAVC)')
plt.ylabel('Number of Individuals')
plt.legend(title='Obesity Category', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# Prepare the dataset for Linear Regression
features = df[['FAF', 'FCVC', 'CH2O', 'TUE', 'MTRANS']]
target = df['BMI']  # Now using the calculated BMI

# One-hot encode the 'MTRANS' column (categorical)
features = pd.get_dummies(features, columns=['MTRANS'], drop_first=True)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nMean Squared Error:", mse)
print("R² Score:", r2)

# Get the coefficients for the features
coefficients = pd.DataFrame(model.coef_, X_train.columns, columns=['Coefficient'])
print("\nModel Coefficients:\n", coefficients)

# Show intercept
print("\nModel Intercept:", model.intercept_)

# Example: Predict BMI for a new person
new_data = {
    'FAF': [1],        # Example values for features
    'FCVC': [2], 
    'CH2O': [2],
    'TUE': [3],
    'MTRANS_Walking': [1],  # Example encoding of 'MTRANS'
    'MTRANS_Bike': [0],     # Other modes of transportation need to be 0 (if not selected)
    'MTRANS_Motorbike': [0],
    'MTRANS_Public_Transportation': [0]
}
# Convert to DataFrame
new_df = pd.DataFrame(new_data)

# Ensure the columns are in the correct order
new_df = new_df[X_train.columns]

# Now make the prediction with the model
predicted_bmi = model.predict(new_df)

# Output the predicted BMI
print("Predicted BMI for the new person:", predicted_bmi[0])
