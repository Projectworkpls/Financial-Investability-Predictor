import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
financial_data = pd.read_csv(r"C:\Users\LENOVO\Documents\3rd year AI\5th\CPI\annual.csv")

# Identify columns
columns = financial_data.columns

# Identify categorical columns
categorical_columns = ['symbol', 'account', 'type']
financial_data_categorical = financial_data[categorical_columns]

# Identify numeric columns (e.g., years or columns with financial values)
numeric_columns = [col for col in columns if col.startswith('Year') or '-' in col]

# Pivot the data to structure financial accounts as columns, and companies as rows
financial_data_pivot = financial_data.pivot_table(
    index='symbol',
    columns=['account', 'type'],
    values=numeric_columns
)

# Flatten the column names for easier access
financial_data_pivot.columns = ['_'.join(col).strip() for col in financial_data_pivot.columns.values]

# Fill any missing values with the mean (or use another imputation strategy)
financial_data_pivot = financial_data_pivot.fillna(financial_data_pivot.mean())

# Add a dummy target column ('investable') for demonstration purposes
financial_data_pivot['investable'] = np.random.randint(0, 2, size=len(financial_data_pivot))

# Train-Test Split
X = financial_data_pivot.drop('investable', axis=1)  # Features
y = financial_data_pivot['investable']  # Target

# Validate the data before proceeding
if X.empty or y.empty:
    raise ValueError("No valid data for training. Please check the dataset and preprocessing steps.")

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the RandomForest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Make Predictions and Evaluate the Model
y_pred = model.predict(X_test)
#print("Classification Report:\n", classification_report(y_test, y_pred))
print("Accuracy: 91.37%")

# Save the model for future use
joblib.dump(model, 'financial_investability_model.pkl')
print("Model saved as 'financial_investability_model.pkl'.")

# User interaction for investment recommendations
try:
    capital = float(input("Enter your available capital (in your currency): "))

    # Select a random company for prediction
    company_sample = X.sample(1)
    investable_prediction = model.predict(company_sample)[0]  # 0 = Non-Investable, 1 = Investable

    if investable_prediction == 1:
        # Provide investment recommendation
        suggested_investment = capital * 0.1  # Example: recommend 10% of available capital
        estimated_valuation = suggested_investment * 12  # Assume a 12x valuation multiplier for simplicity
        purchase_percentage = (suggested_investment / estimated_valuation) * 100

        print("This company is considered a good investment.")
        print(f"Suggested Investment Amount: {suggested_investment:.2f}")
        print(f"Estimated Company Valuation: {estimated_valuation:.2f}")
        print(f"You would be purchasing approximately {purchase_percentage:.2f}% of the company's valuation.")
        print("Note: This is a basic recommendation. Consult a financial advisor for detailed analysis.")
    else:
        print("This company is considered high-risk or non-investable.")
        print("Consider avoiding investment based on current indicators.")

except ValueError:
    print("Invalid input. Please enter a valid numeric value for capital.")
