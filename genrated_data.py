import pandas as pd
import random
import numpy as np
from faker import Faker

# Initialize Faker
fake = Faker()

# Generate synthetic data
def generate_data(num_rows=1000):
    data = []
    categories = ['Toys', 'Snacks', 'Clothing', 'Books', 'Stationery']
    for _ in range(num_rows):
        transaction_id = fake.uuid4()
        product_category = random.choice(categories)
        quantity = random.randint(1, 10)
        price = round(random.uniform(1.99, 50.99), 2)
        customer_age = random.randint(20, 50)
        family_size = random.randint(1, 6)
        date = fake.date_between(start_date='-1y', end_date='today')
        promotion = random.choice(['Yes', 'No'])
        total_sales = round(quantity * price, 2)
        data.append([
            transaction_id, product_category, quantity, price,
            customer_age, family_size, date, promotion, total_sales
        ])
    columns = ['Transaction_ID', 'Product_Category', 'Quantity', 'Price',
               'Customer_Age', 'Family_Size', 'Date', 'Promotion', 'Total_Sales']
    return pd.DataFrame(data, columns=columns)

sales_data = generate_data(1000)

sales_data.to_csv('sales_data.csv', index=False)
print("Sample Data Created!")
