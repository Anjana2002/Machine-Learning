"""
Prepare a dataset of customer having the features date, price, product_id, quantity_purchased,
serial_no, user_id,user_type, user_class, purchase_week and visualise the data with
    a. Plot diagram for Price Trends for Particular User, Price Trends for Particular User Over
    Time
    b. Create box plot Quantity and Week value distribution having parameters of
    quantity_purchased','purchase_week'"""


import pandas as pd
import datetime
import random
import matplotlib.pyplot as plt

num_rcrds=30
dates = pd.date_range(start='2002-02-01', end='2003-10-01',periods=num_rcrds)
prices = [round(random.uniform(10,100),2) for _ in range(num_rcrds)] #rounds the generated number to two decimal places.
product_id = [f'P{random.randint(1,100)}' for  _ in range(num_rcrds)]
quantities = [random.randint(1,10) for  _ in range(num_rcrds)]
serial_no = [f'SN-{random.randint(1000,9999)}'for  _ in range(num_rcrds)]
user_id = ['U' +str(random.randint(10,30))for _ in range(num_rcrds)]
user_type = ['Retail' ,'Wholesale']
user_class = ['Class A', 'Class B', 'Class C']
purchase_week = [date.isocalendar()[1] for date in dates]

data = {
    'Date':dates,
    'Prices':prices,
    'Product_ID': product_id,
    'Quantity_Purchased': quantities,
    'Serial_No': serial_no,
    'User_ID': user_id,
    'User_Type': random.choices(user_type, k=num_rcrds),
    'User_Class': random.choices(user_class, k=num_rcrds),
    'Purchase_Week': purchase_week
}
df= pd.DataFrame(data)
print(df.head())

df.to_csv('customer.csv', index=False)# not to include the index column in the output.
df = pd.read_csv('customer.csv')
user_id='U14'
user_data = df[df['User_ID']==user_id]


# Plot the price trends for the particular user
plt.figure(figsize=(10, 6))
plt.plot(user_data['Date'], user_data['Prices'], alpha=0.5)
plt.xlabel('Date')
plt.ylabel('Prices')
plt.title(f'Price Trends for User ID {user_id}')
plt.xticks(rotation=45)
plt.show()

# Plot the price trends for the particular user over time
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['Prices'], alpha=0.5)
plt.scatter(user_data['Date'], user_data['Prices'], color='red', label=f'User ID {user_id}')
plt.xlabel('Date')
plt.ylabel('Prices')
plt.title(f'Price Trends for User ID {user_id} Over Time')
plt.xticks(rotation=45)
plt.legend()
# Create a box plot for Quantity_Purchased
plt.figure(figsize=(8, 6))
plt.boxplot(df['Quantity_Purchased'])
plt.xlabel('Quantity Purchased')
plt.title('Distribution of Quantity Purchased')
plt.show()

# Create a box plot for Purchase_Week
plt.figure(figsize=(8, 6))
plt.boxplot(df['Purchase_Week'])
plt.xlabel('Purchase Week')
plt.title('Distribution of Purchase Week')
plt.show()
