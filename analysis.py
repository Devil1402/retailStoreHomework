import pandas as pd
import plotly.express as px
import plotly.io as pio

# Set Plotly JSON engine to avoid orjson issues
pio.json.config.engine = 'json'

# Function to calculate summary metrics for the dataset
def summary_metrics(data):
    """
    Calculates and returns key summary metrics based on the dataset.

    Args:
        data (pd.DataFrame): Input dataset containing purchase data.

    Returns:
        dict: A dictionary containing total revenue, average revenue per unit,
              top category, best-selling product, and most profitable product.
    """
    total_revenue = data['Purchase_Amount'].sum()
    avg_revenue_per_unit = data['Purchase_Amount'].sum() / data['Purchase_Quantity'].sum()
    top_category = data.groupby('Category')['Purchase_Amount'].sum().idxmax()
    
    # Identify the best-selling product
    best_selling_product = (
        data.groupby(['Product_ID', 'Subcategory'])
        .agg({'Purchase_Quantity': 'sum'})
        .reset_index()
        .sort_values(by='Purchase_Quantity', ascending=False)
        .iloc[0]
    )

    # Identify the most profitable product
    most_profitable_product = (
        data.groupby(['Product_ID', 'Subcategory'])
        .apply(lambda x: x['Purchase_Amount'].sum() / x['Purchase_Quantity'].sum())
        .reset_index(name='Avg_Revenue_Per_Unit')
        .sort_values(by='Avg_Revenue_Per_Unit', ascending=False)
        .iloc[0]
    )

    return {
        'Total Revenue': f"${total_revenue:,.2f}",
        'Avg Revenue Per Unit': f"${avg_revenue_per_unit:.2f}",
        'Top Category': top_category,
        'Best Selling Product': f"{best_selling_product['Product_ID']} ({best_selling_product['Subcategory']})",
        'Most Profitable Product': f"{most_profitable_product['Product_ID']} ({most_profitable_product['Subcategory']})"
    }

# Function to analyze product profitability
def product_profitability_analysis(data, selected_category=None):
    """
    Creates a scatter plot for product profitability analysis based on purchase data.

    Args:
        data (pd.DataFrame): Input dataset containing purchase data.
        selected_category (str, optional): Category to filter the data. Defaults to None.

    Returns:
        tuple: A plotly figure object and a summary string.
    """
    if selected_category:
        data = data[data['Category'] == selected_category]

    # Aggregate sales data by product and subcategory
    product_sales = data.groupby(['Product_ID', 'Subcategory']).agg({
        'Purchase_Quantity': 'sum',
        'Purchase_Amount': 'sum'
    }).reset_index()

    product_sales['Avg_Revenue_Per_Unit'] = product_sales['Purchase_Amount'] / product_sales['Purchase_Quantity']
    product_sales['Product_Label'] = product_sales['Product_ID'].astype(str) + " (" + product_sales['Subcategory'] + ")"

    # Create scatter plot
    fig = px.scatter(
        product_sales,
        x='Purchase_Quantity',
        y='Purchase_Amount',
        size='Avg_Revenue_Per_Unit',
        color='Avg_Revenue_Per_Unit',
        hover_name='Product_Label',
        title=f'Product Profitability Analysis{" - " + selected_category if selected_category else ""}',
        labels={
            'Purchase_Quantity': 'Units Sold',
            'Purchase_Amount': 'Total Revenue ($)',
            'Avg_Revenue_Per_Unit': 'Avg Revenue per Unit ($)',
            'Product_Label': 'Product'
        },
        color_continuous_scale='Viridis',
        size_max=30
    )
    fig.update_layout(
        xaxis_title='Units Sold',
        yaxis_title='Total Revenue ($)',
        legend_title='Avg Revenue per Unit ($)'
    )

    # Generate dynamic summary
    max_revenue_product = product_sales.loc[product_sales['Purchase_Amount'].idxmax()]
    summary = (
        f"The product '{max_revenue_product['Product_Label']}' generated the highest revenue of ${max_revenue_product['Purchase_Amount']:,.2f}. "
        f"The average revenue per unit for this product is ${max_revenue_product['Avg_Revenue_Per_Unit']:.2f}. "
        f"This indicates strong performance in terms of sales and profitability."
    )

    return fig, summary

# Function to analyze sales by category
def category_sales_analysis(data):
    """
    Creates a pie chart for category sales distribution.

    Args:
        data (pd.DataFrame): Input dataset containing purchase data.

    Returns:
        tuple: A plotly figure object and a summary string.
    """
    category_sales = data.groupby('Category').agg({'Purchase_Amount': 'sum'}).reset_index()

    fig = px.pie(
        category_sales,
        names='Category',
        values='Purchase_Amount',
        title='Category Sales Distribution',
        hole=0.4,
        color_discrete_sequence=px.colors.sequential.RdBu
    )

    # Generate summary for top category
    top_category = category_sales.loc[category_sales['Purchase_Amount'].idxmax()]
    summary = (
        f"The category '{top_category['Category']}' accounts for the highest sales, contributing ${top_category['Purchase_Amount']:,.2f} to total revenue. "
        "This highlights a key area for further investment and promotion."
    )

    return fig, summary

# Function to analyze subcategory performance
def subcategory_analysis(data):
    """
    Creates a bar chart for subcategory sales analysis.

    Args:
        data (pd.DataFrame): Input dataset containing purchase data.

    Returns:
        tuple: A plotly figure object and a summary string.
    """
    subcategory_sales = data.groupby('Subcategory').agg({
        'Purchase_Amount': 'sum',
        'Purchase_Quantity': 'sum'
    }).reset_index()

    fig = px.bar(
        subcategory_sales,
        x='Subcategory',
        y='Purchase_Amount',
        title='Subcategory Sales Analysis',
        labels={'Purchase_Amount': 'Total Revenue ($)', 'Subcategory': 'Subcategory'},
        text='Purchase_Amount',
        color='Purchase_Amount',
        color_continuous_scale='Blues'
    )
    fig.update_traces(texttemplate='%{text:.2s}', textposition='outside')

    # Generate summary for top subcategory
    top_subcategory = subcategory_sales.loc[subcategory_sales['Purchase_Amount'].idxmax()]
    summary = (
        f"The subcategory '{top_subcategory['Subcategory']}' leads in revenue with ${top_subcategory['Purchase_Amount']:,.2f}, "
        f"selling a total of {top_subcategory['Purchase_Quantity']} units."
    )

    return fig, summary

# Function to analyze customer-specific data
def user_analysis(data, selected_customer_id):
    """
    Creates a bar chart for user-specific analysis and provides a summary.

    Args:
        data (pd.DataFrame): Input dataset containing purchase data.
        selected_customer_id (int): Customer ID for analysis.

    Returns:
        tuple: A plotly figure object and a summary string.

    Raises:
        ValueError: If 'Customer_ID' column is not in the dataset.
    """
    if 'Customer_ID' not in data.columns:
        raise ValueError("The dataset does not contain a 'Customer_ID' column.")

    customer_data = data[data['Customer_ID'] == selected_customer_id]
    total_spending = customer_data['Purchase_Amount'].sum()
    avg_spending = customer_data['Purchase_Amount'].mean()
    total_quantity = customer_data['Purchase_Quantity'].sum()

    metrics = {
        'Metric': ['Total Spending', 'Average Spending', 'Total Quantity Purchased'],
        'Value': [total_spending, avg_spending, total_quantity]
    }
    metrics_df = pd.DataFrame(metrics)

    # Create bar chart for customer analysis
    fig = px.bar(
        metrics_df,
        x='Metric',
        y='Value',
        text='Value',
        color='Metric',
        title=f"Customer Analysis - Customer ID: {selected_customer_id}",
        labels={'Value': 'Amount', 'Metric': 'Metrics'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_traces(
        texttemplate='%{text:.2f}',
        textposition='outside',
        marker=dict(line=dict(color='black', width=1))
    )
    fig.update_layout(
        yaxis=dict(title='Amount ($)', gridcolor='lightgrey'),
        xaxis=dict(title='Metrics', tickangle=0),
        title=dict(
            text=f"<b>Customer Analysis - Customer ID: {selected_customer_id}</b>",
            x=0.5,
            xanchor='center',
            font=dict(size=20)
        ),
        font=dict(size=14),
        plot_bgcolor='white',
        showlegend=False
    )

    # Generate summary for the selected customer
    summary = (
        f"Customer {selected_customer_id} has spent a total of ${total_spending:,.2f} across {total_quantity} items purchased. "
        f"Their average spending per purchase is ${avg_spending:,.2f}."
    )

    return fig, summary
