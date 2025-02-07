import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('cleaned_sales_data.csv')

sns.set(style="whitegrid")

# 1. Visualize the Distribution of 'Price'
plt.figure(figsize=(8, 6))
sns.histplot(data['Price'], kde=True, color='blue', bins=30)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# 2. Visualize the Distribution of 'Total_Sales'
plt.figure(figsize=(8, 6))
sns.histplot(data['Total_Sales'], kde=True, color='green', bins=30)
plt.title('Total Sales Distribution')
plt.xlabel('Total Sales')
plt.ylabel('Frequency')
plt.show()

# 3. Analyze Relationship Between 'Price' and 'Total_Sales'
plt.figure(figsize=(8, 6))
sns.scatterplot(x=data['Price'], y=data['Total_Sales'], color='red')
plt.title('Price vs Total Sales')
plt.xlabel('Price')
plt.ylabel('Total Sales')
plt.show()

# 4. Boxplot for 'Price' by 'Product_Category'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Product_Category_Clothing', y='Price', data=data)
plt.title('Price Distribution by Product Category')
plt.xlabel('Product Category')
plt.ylabel('Price')
plt.show()


# 5. Correlation Matrix for Numerical Variables
correlation_matrix = data[['Price', 'Total_Sales', 'Month']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# 6. Total Sales by 'Month'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Month', y='Total_Sales', data=data)
plt.title('Total Sales by Month')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.show()

# 7. Total Sales by 'Season'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Season', y='Total_Sales', data=data)
plt.title('Total Sales by Season')
plt.xlabel('Season')
plt.ylabel('Total Sales')
plt.show()

# 8. Boxplot for 'Promotion' vs 'Total_Sales'
plt.figure(figsize=(8, 6))
sns.boxplot(x='Promotion_Yes', y='Total_Sales', data=data)
plt.title('Total Sales with and without Promotion')
plt.xlabel('Promotion')
plt.ylabel('Total Sales')
plt.show()
print(data.columns)
