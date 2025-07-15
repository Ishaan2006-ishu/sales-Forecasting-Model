import pandas as pd

train = pd.read_csv('train.csv')
features = pd.read_csv('features.csv')
stores = pd.read_csv('stores.csv')
test=pd.read_csv('test.csv')



# Example: Drop columns with too many missing values
train = train.drop(['MarkDown2', 'MarkDown5'], axis=1, errors='ignore')
features = features.drop(['MarkDown2', 'MarkDown5'], axis=1, errors='ignore')

train['Date'] = pd.to_datetime(train['Date'])
features['Date'] = pd.to_datetime(features['Date'])

# Merge train with stores and features
df = train.merge(stores, on='Store', how='left')
df = df.merge(features, on=['Store', 'Date'], how='left')



#You either fill the missing data or remove the column.
df.fillna(0, inplace=True)

df['Year']=df['Date'].dt.year
df['Month']=df['Date'].dt.month
df['Week']=df['Date'].dt.isocalendar().week
df['Day']=df['Date'].dt.day

# This will create new columns: Type_B and Type_C (Type_A is dropped)
df = pd.get_dummies(df, columns=['Type'], drop_first=True)

df.head()

df['IsHoliday_x'] = df['IsHoliday_x'].astype(int)


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 5))
sns.lineplot(data=df, x='Month', y='Weekly_Sales', estimator='sum')
plt.title("Total Weekly Sales by Month")
plt.ylabel("Total Sales")
plt.show()

df.head()

plt.figure(figsize=(8, 5))
sns.boxplot(x='Type_B', y='Weekly_Sales', data=df)
plt.title("Sales Distribution for Store Type B")
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# Drop columns we won't use for input
X = df.drop(columns=['Weekly_Sales', 'Date'])

# Target column we want to predict
y = df['Weekly_Sales']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test=train_test_split(
    X,y, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestRegressor
model=RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

from sklearn.metrics import mean_squared_error

rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Take the square root manually
print(f"rmse:{rmse:.2f}")


from sklearn.metrics import r2_score

r2 = r2_score(y_test, y_pred)

print(f"R² Score: {r2:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Weekly Sales")
plt.show()


importances = model.feature_importances_
feature_names = X.columns
feature_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feature_df[:10], y=feature_df.index[:10])
plt.title("Top 10 Important Features")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

errors = y_test - y_pred

plt.figure(figsize=(10, 5))
sns.histplot(errors, bins=30, kde=True)
plt.title("Distribution of Prediction Errors")
plt.xlabel("Error (Actual - Predicted)")
plt.ylabel("Frequency")
plt.show()

from sklearn.linear_model import LinearRegression

# Step 1: Create model
lin_model = LinearRegression()

# Step 2: Train model
lin_model.fit(X_train, y_train)

# Step 3: Predict
lin_pred = lin_model.predict(X_test)

# Step 4: Evaluate
from sklearn.metrics import mean_squared_error, r2_score

rmse_lin = mean_squared_error(y_test, lin_pred)**0.5
r2_lin = r2_score(y_test, lin_pred)

print(f"Linear Regression RMSE: {rmse_lin:.2f}")
print(f"Linear Regression R² Score: {r2_lin:.2f}")

import joblib

# Save model
joblib.dump(model, 'random_forest_model.pkl')

# To load later:
# model = joblib.load('random_forest_model.pkl')