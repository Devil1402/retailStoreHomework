import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import date, timedelta

# Initialize for synthetic data generation
random.seed(42)
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Constants
NUM_CUSTOMERS = 500  # Total number of customers
NUM_PRODUCTS = 50    # Total number of products
NUM_PURCHASES = 30000  # Target number of purchase records

# Product categories and their corresponding subcategories
PRODUCT_CATEGORIES = {
    'Electronics': ['Smartphones', 'Laptops', 'Headphones', 'Tablets', 'Accessories'],
    'Clothing': ['Shirts', 'Pants', 'Dresses', 'Shoes', 'Accessories'],
    'Home & Living': ['Furniture', 'Kitchen', 'Decor', 'Bedding', 'Storage'],
    'Books': ['Fiction', 'Non-Fiction', 'Educational', 'Comics', 'Magazines'],
    'Beauty': ['Skincare', 'Makeup', 'Haircare', 'Fragrances', 'Tools']
}

def generate_product_data():
    """
    Generates synthetic product data with categories, subcategories, base prices, 
    and popularity scores.

    Returns:
    - pd.DataFrame: DataFrame containing product data.
    """
    products = []
    product_id = 1  # Unique identifier for products
    
    # Generate products for each category and subcategory
    for category in PRODUCT_CATEGORIES:
        for subcategory in PRODUCT_CATEGORIES[category]:
            base_price = {
                'Electronics': random.uniform(100, 800),
                'Clothing': random.uniform(10, 150),
                'Home & Living': random.uniform(30, 400),
                'Books': random.uniform(5, 40),
                'Beauty': random.uniform(10, 100)
            }[category]
            
            products.append({
                'Product_ID': product_id,
                'Category': category,
                'Subcategory': subcategory,
                'Base_Price': round(base_price, 2),
                'Popularity_Score': random.uniform(0.1, 1.0)
            })
            product_id += 1
    
    # Ensure the total number of products matches NUM_PRODUCTS
    while product_id <= NUM_PRODUCTS:
        category = random.choice(list(PRODUCT_CATEGORIES.keys()))
        subcategory = random.choice(PRODUCT_CATEGORIES[category])
        base_price = {
            'Electronics': random.uniform(100, 800),
            'Clothing': random.uniform(10, 150),
            'Home & Living': random.uniform(30, 400),
            'Books': random.uniform(5, 40),
            'Beauty': random.uniform(10, 100)
        }[category]
        
        products.append({
            'Product_ID': product_id,
            'Category': category,
            'Subcategory': subcategory,
            'Base_Price': round(base_price, 2),
            'Popularity_Score': random.uniform(0.1, 1.0)
        })
        product_id += 1
    
    return pd.DataFrame(products)

def generate_customer_data():
    """
    Generates synthetic customer data, including spending power, purchase frequency, 
    and category preferences.

    Returns:
    - pd.DataFrame: DataFrame containing customer data.
    """
    customers = []
    for cid in range(1, NUM_CUSTOMERS + 1):
        # Generate category preferences using Dirichlet distribution
        category_preferences = np.random.dirichlet(np.ones(len(PRODUCT_CATEGORIES)))
        spending_power = np.random.choice(['Low', 'Medium', 'High'], p=[0.2, 0.6, 0.2])
        purchase_frequency = np.random.choice(['Rare', 'Regular', 'Frequent'], p=[0.3, 0.5, 0.2])
        
        customers.append({
            'Customer_ID': cid,
            'Spending_Power': spending_power,
            'Purchase_Frequency': purchase_frequency,
            **{f'Preference_{cat}': pref for cat, pref in zip(PRODUCT_CATEGORIES.keys(), category_preferences)}
        })
    return pd.DataFrame(customers)

def generate_purchase_data(customers_df, products_df):
    """
    Generates synthetic purchase data by simulating customer-product interactions.

    Parameters:
    - customers_df (pd.DataFrame): DataFrame containing customer data.
    - products_df (pd.DataFrame): DataFrame containing product data.

    Returns:
    - pd.DataFrame: DataFrame containing purchase records.
    """
    purchases = []
    customer_purchase_count = {cid: 0 for cid in customers_df['Customer_ID']}
    
    # Generate dates for the last 2 years
    start_date = date.today() - timedelta(days=730)
    dates = [start_date + timedelta(days=x) for x in range(730)]
    
    for _ in range(NUM_PURCHASES):
        customer = customers_df.sample(n=1).iloc[0]
        
        # Preferentially select the last purchased category with a probability of 30%
        if customer_purchase_count[customer['Customer_ID']] > 0 and random.random() < 0.3:
            last_category = purchases[-1]['Category']
            product = products_df[products_df['Category'] == last_category].sample(n=1).iloc[0]
        else:
            product = products_df.sample(n=1).iloc[0]
        
        # Determine purchase quantity based on spending power
        quantity_probs = {
            'Low': [0.9, 0.1, 0],
            'Medium': [0.7, 0.2, 0.1],
            'High': [0.5, 0.3, 0.2]
        }[customer['Spending_Power']]
        purchase_quantity = np.random.choice([1, 2, 3], p=quantity_probs)
        
        # Calculate the final purchase amount
        base_amount = product['Base_Price'] * purchase_quantity
        spending_variability = np.random.uniform(0.85, 1.15)
        final_amount = base_amount * spending_variability
        
        # Randomly assign a purchase date
        purchase_date = random.choice(dates)
        
        purchases.append({
            'Customer_ID': customer['Customer_ID'],
            'Product_ID': product['Product_ID'],
            'Category': product['Category'],
            'Subcategory': product['Subcategory'],
            'Purchase_Amount': round(final_amount, 2),
            'Purchase_Quantity': purchase_quantity,
            'Unit_Price': round(final_amount / purchase_quantity, 2),
            'Purchase_Date': purchase_date
        })
        
        customer_purchase_count[customer['Customer_ID']] += 1
    
    return pd.DataFrame(purchases)

# Generate synthetic datasets
products_df = generate_product_data()
customers_df = generate_customer_data()
purchases_df = generate_purchase_data(customers_df, products_df)

# Create the final dataset with additional metrics
final_data = purchases_df.copy()

# Add derived metrics for customers
final_data['Customer_Purchase_Count'] = final_data.groupby('Customer_ID').cumcount() + 1
final_data['Customer_Total_Spent'] = final_data.groupby('Customer_ID')['Purchase_Amount'].transform('cumsum')
final_data['Customer_Average_Order'] = final_data.groupby('Customer_ID')['Purchase_Amount'].transform('mean')

# Remove duplicate records
final_data = final_data.drop_duplicates()

# Save the dataset to a CSV file
output_file = "customerPurchaseData.csv"
final_data.to_csv(output_file, index=False)

print(f"Enhanced synthetic dataset with {len(final_data)} records saved to '{output_file}'")

# Print dataset statistics
print("\nDataset Statistics:")
print(f"Number of unique customers: {final_data['Customer_ID'].nunique()}")
print(f"Number of unique products: {final_data['Product_ID'].nunique()}")
print(f"Date range: {min(final_data['Purchase_Date'])} to {max(final_data['Purchase_Date'])}")
print(f"Average purchase amount: ${final_data['Purchase_Amount'].mean():.2f}")