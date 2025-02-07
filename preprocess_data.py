import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('sales_data.csv')

print(data.head())
print(data.info())
print(data.isnull().sum())

data = data.dropna()  

data['Date'] = pd.to_datetime(data['Date'])

data['Day_of_Week'] = data['Date'].dt.day_name()
data['Month'] = data['Date'].dt.month

def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

data['Season'] = data['Month'].apply(get_season)

scaler = MinMaxScaler()
data[['Price', 'Total_Sales']] = scaler.fit_transform(data[['Price', 'Total_Sales']])

data = pd.get_dummies(data, columns=['Product_Category', 'Promotion'], drop_first=True)

data.to_csv("cleaned_sales_data.csv", index=False)

print(data.head())
print(data.dtypes)
# Drop non-numeric columns 
data = data.select_dtypes(include=[float, int])

