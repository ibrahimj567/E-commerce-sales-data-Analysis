import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

data = pd.read_csv('cleaned_sales_data.csv')
data = data.select_dtypes(include=[float, int])

X = data.drop(columns=['Total_Sales']) 
y = data['Total_Sales']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_split=2, min_samples_leaf=1, random_state=42)

rf_model.fit(X_train, y_train)

joblib.dump(rf_model, 'sales_predictor_model.pkl')

print("Model saved successfully!")
