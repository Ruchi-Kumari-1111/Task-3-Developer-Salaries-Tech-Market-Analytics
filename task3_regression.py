import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Set visualization style for professional graphs
sns.set_theme(style="whitegrid", palette="muted")

def print_separator(title):
    print("\n" + "=" * 80)
    print(f" {title.upper()} ")
    print("=" * 80)

def print_section(title):
    print("\n--- " + title + " ---")

# ==========================================
# 1. IMPORT & PREPROCESS THE DATASET
# ==========================================
print_separator("1. Data Preprocessing & Generation")
print("Generating synthetic developer salary dataset based on Experience, Languages, Meetings, and Remote Work...")

np.random.seed(42)
n_samples = 300

experience = np.random.uniform(1, 20, n_samples)
languages = np.random.randint(1, 7, n_samples)
meetings = np.random.uniform(2, 25, n_samples)
remote = np.random.uniform(0, 100, n_samples)

# True mathematical relationship with some added noise
noise = np.random.normal(0, 6000, n_samples)
salary = 40000 + (5000 * experience) + (2000 * languages) - (200 * meetings) + (50 * remote) + noise

df = pd.DataFrame({
    'YearsExperience': experience,
    'KnownLanguages': languages,
    'MeetingHours': meetings,
    'RemotePercent': remote,
    'Salary': salary
})

print("\nDataset successfully generated. Here is a preview of the raw data:\n")
print(df.head())
print("\nDataset contains 300 samples with no null values. Preprocessing complete.")


# ==========================================
# 2. SIMPLE LINEAR REGRESSION
# ==========================================
print_separator("2. Simple Linear Regression (Salary vs. Experience)")

# Split data
X_simple = df[['YearsExperience']]
y_simple = df['Salary']
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

print("Train-test split complete: 80% Training, 20% Testing.")

# Fit model
model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)
y_pred_s = model_simple.predict(X_test_s)

# Metrics
mae_s = mean_absolute_error(y_test_s, y_pred_s)
mse_s = mean_squared_error(y_test_s, y_pred_s)
r2_s = r2_score(y_test_s, y_pred_s)

print("\nModel Training Completed.")
print("\nModel Performance Metrics:")
print(f" * Mean Absolute Error (MAE): {mae_s:.2f}")
print(f" * Mean Squared Error (MSE):  {mse_s:.2f}")
print(f" * R-squared Score (R2):      {r2_s:.4f}")

print("\nFeature Coefficients:")
print(f" * YearsExperience: {model_simple.coef_[0]:.6f}")
print(f" * Intercept:       {model_simple.intercept_:.6f}")

print_section("Interpretation & Graph Explanation")
print("""
What this means in plain terms:
- The coefficient for YearsExperience is roughly ~5000. This indicates that for every 
  single additional year of experience a developer gains, their predicted annual salary 
  increases by about $5,000, assuming no other factors are considered.
- The intercept represents the theoretical baseline starting salary for someone with 
  zero years of experience in this specific model.

What you will see in the upcoming graph:
- A scatterplot featuring blue dots representing the testing data (actual developer 
  salaries plotted against their years of experience).
- A solid red line passing through the center of the data cluster. This is the 
  "line of best fit." It represents our mathematical model's attempt to draw a 
  relationship between experience and money with the lowest possible error.
""")

print(">>> Close the graph window to continue to the Multiple Linear Regression section. <<<")

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_test_s['YearsExperience'], y=y_test_s, color='blue', alpha=0.6, label='Actual Data')
sns.lineplot(x=X_test_s['YearsExperience'], y=y_pred_s, color='red', linewidth=2, label='Regression Line')
plt.title('Simple Linear Regression: Salary vs. Years of Experience', fontsize=14, fontweight='bold')
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Annual Salary ($)', fontsize=12)
plt.legend()
plt.show()


# ==========================================
# 3. MULTIPLE LINEAR REGRESSION
# ==========================================
print_separator("3. Multiple Linear Regression (All Features)")

# Split data
X_multi = df[['YearsExperience', 'KnownLanguages', 'MeetingHours', 'RemotePercent']]
y_multi = df['Salary']
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

print("Train-test split complete: 80% Training, 20% Testing using all 4 features.")

# Fit model
model_multi = LinearRegression()
model_multi.fit(X_train_m, y_train_m)
y_pred_m = model_multi.predict(X_test_m)

# Metrics
mae_m = mean_absolute_error(y_test_m, y_pred_m)
mse_m = mean_squared_error(y_test_m, y_pred_m)
r2_m = r2_score(y_test_m, y_pred_m)

print("\nModel Training Completed.")
print("\nModel Performance Metrics:")
print(f" * Mean Absolute Error (MAE): {mae_m:.2f}")
print(f" * Mean Squared Error (MSE):  {mse_m:.2f}")
print(f" * R-squared Score (R2):      {r2_m:.4f}")

print("\nFeature Coefficients:")
for feature, coef in zip(X_multi.columns, model_multi.coef_):
    print(f" * {feature:<16}: {coef:.6f}")
print(f" * Intercept       : {model_multi.intercept_:.6f}")

print_section("Interpretation & Graph Explanation")
print("""
By extracting the model's coefficients, we derive actual business logic from the mathematics, 
isolating the effect of one variable while holding the others constant:

- YearsExperience (~5000) : Holding all else equal, an extra year of experience adds $5,000.
- KnownLanguages  (~2000) : Learning an additional programming language adds $2,000.
- MeetingHours    (~ -200): This coefficient is negative. For every additional hour spent 
                            in meetings per week, the predicted salary drops by $200.
- RemotePercent   (~50)   : Every 1% increase in remote work capability boosts predicted salary by $50.

Metric Evaluation Comparison:
- MAE (Mean Absolute Error): Shows the average absolute dollar amount the model is off by. 
- MSE (Mean Squared Error) : Punishes larger errors much heavier than smaller ones.
- R-squared Score          : Explains the variance. A higher R2 score here means that predicting 
                             with all four variables explains the variance in a developer's salary 
                             much better than using experience alone.

What you will see in the upcoming graph:
- Because we are predicting using four dimensions, we cannot draw a simple line on a 2D graph.
- Instead, you will see a scatter plot comparing Actual Salaries (X-axis) against 
  Predicted Salaries (Y-axis). 
- There is a dotted black diagonal line representing "Perfect Prediction" (Actual = Predicted). 
  The closer our purple data points cluster to this diagonal line, the more accurate the model is.
""")

print(">>> Displaying final graph. Script will exit when graph is closed. <<<")

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test_m, y=y_pred_m, color='purple', alpha=0.7)
plt.plot([y_test_m.min(), y_test_m.max()], [y_test_m.min(), y_test_m.max()], color='black', linestyle='--', linewidth=2, label='Perfect Prediction')
plt.title('Multiple Linear Regression: Actual vs. Predicted Salaries', fontsize=14, fontweight='bold')
plt.xlabel('Actual Salary ($)', fontsize=12)
plt.ylabel('Predicted Salary ($)', fontsize=12)
plt.legend()
plt.show()