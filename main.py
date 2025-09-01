import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
# Generate synthetic HR data
n_employees = 500
data = {
    'EmployeeID': range(1, n_employees + 1),
    'Age': np.random.randint(20, 60, size=n_employees),
    'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'], size=n_employees),
    'YearsExperience': np.random.randint(0, 20, size=n_employees),
    'Salary': np.random.randint(40000, 150000, size=n_employees),
    'SatisfactionScore': np.random.normal(loc=5, scale=2, size=n_employees).clip(0, 10).round(1), #satisfaction score between 0 and 10
    'WorkLifeBalance': np.random.normal(loc=5, scale=2, size=n_employees).clip(0,10).round(1), #work life balance between 0 and 10
    'LeftCompany': np.random.binomial(1, 0.15, size=n_employees) # 15% turnover rate
}
df = pd.DataFrame(data)
# --- 2. Data Cleaning and Preparation ---
#Handle missing values (simulated here)
df['SatisfactionScore'] = df['SatisfactionScore'].fillna(df['SatisfactionScore'].mean())
df['WorkLifeBalance'] = df['WorkLifeBalance'].fillna(df['WorkLifeBalance'].mean())
# Feature Engineering
df['YearsExperience_Squared'] = df['YearsExperience']**2
df['Salary_Category'] = pd.cut(df['Salary'], bins=[0, 60000, 100000, 1000000], labels=['Low', 'Medium', 'High'], right=False)
# --- 3. Analysis ---
# Descriptive Statistics
print("\nDescriptive Statistics:")
print(df.describe())
# Turnover Rate by Department
turnover_by_dept = df.groupby('Department')['LeftCompany'].mean()
print("\nTurnover Rate by Department:")
print(turnover_by_dept)
# --- 4. Visualization ---
plt.figure(figsize=(12, 6))
sns.histplot(df['SatisfactionScore'], kde=True, stat="density", kde_kws=dict(cut=3), color='skyblue', label='Satisfaction Score')
sns.histplot(df['WorkLifeBalance'], kde=True, stat="density", kde_kws=dict(cut=3), color='coral', label='Work Life Balance')
plt.title('Distribution of Satisfaction and Work Life Balance')
plt.xlabel('Score (0-10)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('satisfaction_worklifebalance_distribution.png')
print("Plot saved to satisfaction_worklifebalance_distribution.png")
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='SatisfactionScore', data=df, hue='LeftCompany')
plt.title('Satisfaction Score by Department and Turnover')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('satisfaction_by_department_turnover.png')
print("Plot saved to satisfaction_by_department_turnover.png")
#Correlation Matrix
corr_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")