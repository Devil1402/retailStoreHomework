import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px

def customer_segmentation(data):
    """
    Performs customer segmentation based on shopping behavior and spending patterns.

    Args:
        data (pd.DataFrame): Input dataset containing customer purchase information.

    Returns:
        tuple: A plotly scatter plot figure for visualization and a dictionary containing
               dynamic summaries for each customer segment.
    """
    # Step 1: Aggregate customer data
    customer_data = data.groupby('Customer_ID').agg({
        'Purchase_Amount': ['sum', 'mean'],
        'Purchase_Quantity': ['sum'],
        'Purchase_Date': 'nunique'
    }).reset_index()

    # Rename columns for clarity
    customer_data.columns = [
        'Customer_ID', 'Total_Spending', 'Avg_Spending',
        'Total_Quantity', 'Shopping_Frequency'
    ]

    # Calculate average spending per item and handle division by zero
    customer_data['Avg_Spend_Per_Item'] = customer_data['Total_Spending'] / customer_data['Total_Quantity']
    customer_data['Avg_Spend_Per_Item'] = customer_data['Avg_Spend_Per_Item'].fillna(0)

    # Step 2: Scale features for clustering
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(customer_data[['Total_Spending', 'Shopping_Frequency', 'Avg_Spend_Per_Item']])

    # Step 3: Apply KMeans clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    customer_data['Cluster'] = kmeans.fit_predict(scaled_features)

    # Map clusters to descriptive segment names
    cluster_labels = {
        0: "Value Seekers",
        1: "Frequent Shoppers",
        2: "Occasional Buyers",
        3: "High Spenders"
    }
    customer_data['Customer Segment'] = customer_data['Cluster'].map(cluster_labels)

    # Define cluster annotations for the visualization
    cluster_annotations = {
        "Frequent Shoppers": (40, 10),
        "High Spenders": (50, 18),
        "Value Seekers": (60, 14),
        "Occasional Buyers": (70, 22)
    }

    # Step 4: Create scatter plot for segmentation visualization
    fig = px.scatter(
        customer_data,
        x='Shopping_Frequency',
        y='Total_Spending',
        color='Customer Segment',
        size='Avg_Spend_Per_Item',
        size_max=15,
        template='plotly_white',
        title="Customer Segmentation Analysis",
        labels={
            'Shopping_Frequency': '← Less Frequent Shopping | More Frequent Shopping →',
            'Total_Spending': '↓ Lower Spending | Higher Spending ↑'
        }
    )

    # Add subtitle to the plot
    fig.update_layout(
        title={
            'text': "Customer Segmentation Analysis<br><sup>Based on Shopping Behavior and Spending Patterns</sup>",
            'x': 0.5,
            'xanchor': 'center'
        }
    )

    # Annotate clusters for clarity in the visualization
    for segment, (x, y) in cluster_annotations.items():
        fig.add_annotation(
            x=x,
            y=y,
            text=segment,
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            bgcolor="white",
            font=dict(size=12, color="black")
        )

    # Step 5: Generate summaries for each customer segment
    segment_summaries = {}
    for segment, label in cluster_labels.items():
        segment_data = customer_data[customer_data['Customer Segment'] == label]

        total_customers = len(segment_data)
        avg_spending = segment_data['Total_Spending'].mean()
        avg_frequency = segment_data['Shopping_Frequency'].mean()
        avg_spend_per_item = segment_data['Avg_Spend_Per_Item'].mean()

        summary = (
            f"The '{label}' segment has {total_customers} customers. On average, customers in this segment spend "
            f"${avg_spending:,.2f} across {avg_frequency:.1f} shopping trips, with an average spend per item of "
            f"${avg_spend_per_item:,.2f}. This indicates that {label.lower()} tend to exhibit {'higher' if avg_spending > customer_data['Total_Spending'].mean() else 'lower'} "
            f"spending patterns compared to other segments."
        )
        segment_summaries[label] = summary

    # Return the plot and the segment summaries
    return fig, segment_summaries
