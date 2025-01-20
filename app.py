import streamlit as st
import pandas as pd
import plotly.io as pio
from analysis import summary_metrics, product_profitability_analysis, category_sales_analysis, subcategory_analysis, user_analysis
from segmentation import customer_segmentation
from recommender_system import train_and_save_model, get_recommendations_for_user
from config import DATA_PATH

# Set Plotly JSON engine to avoid orjson issues
pio.json.config.engine = 'json'

st.set_page_config(
    page_title="Customer Insights Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    body {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        background-color: #f4f7fb;
    }

    .main-title {
        font-size: 42px;
        font-weight: bold;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: bold;
        color: #34495e;
        margin-top: 40px;
        margin-bottom: 10px;
        border-left: 6px solid #2c3e50;
        padding-left: 10px;
    }

    .metric-card {
        background-color: #ffffff;
        border: 2px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        text-align: center;
        font-size: 18px;
        font-weight: bold;
        color: #2c3e50;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }

    .recommendation-box {
        background-color: #f9f9f9;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        font-size: 16px;
        color: #34495e;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }

    .summary-box {
        background-color: #eef7f9;
        border: 1px solid #cce5e7;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 20px;
        font-size: 16px;
        color: #34495e;
    }

    .clustered-summary {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        gap: 15px;
    }

    .summary-item {
        flex: 1 1 calc(48% - 10px);
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        color: #2c3e50;
        box-shadow: 1px 1px 3px rgba(0, 0, 0, 0.1);
    }

    .summary-icon {
        font-size: 32px;
        margin-right: 10px;
        color: #2980b9;
    }

    .report-section {
        background-color: #ffffff;
        border: 1px solid #dddddd;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 1px 1px 5px rgba(0, 0, 0, 0.1);
    }

    .report-icon {
        font-size: 24px;
        color: #16a085;
        margin-right: 10px;
    }

    .recommendation-section {
        background-color: #fdfdfd;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
    }

    .recommendation-title {
        font-size: 20px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 15px;
    }

    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-title">Customer Insights Dashboard</h1>', unsafe_allow_html=True)

data = None
try:
    data = pd.read_csv(DATA_PATH)
    data['Purchase_Date'] = pd.to_datetime(data['Purchase_Date'])
except FileNotFoundError:
    st.error(f"Dataset not found at {DATA_PATH}. Please ensure the path is correct in config.py.")
    st.stop()

def analysis_page(data):
    """
    Renders the dashboard overview page with filters, key metrics, and various analyses.

    Parameters:
    - data (pd.DataFrame): The dataset containing purchase information.

    This function includes several sections:
    1. Filters Section: Allows users to filter data by date and category.
    2. Key Metrics Section: Displays summary metrics like total revenue and top category.
    3. Product Profitability Analysis: Analyzes the profitability of products.
    4. Category Sales Distribution: Visualizes sales distribution across categories.
    5. Subcategory Sales Analysis: Provides insights into subcategory performance.
    6. Customer-Level Analysis: Enables customer-specific analysis (if available in the dataset).
    """
    # Dashboard Title
    st.markdown('<div class="section-title">Dashboard Overview</div>', unsafe_allow_html=True)

    # Filters Section: Enables data filtering by date range and category
    st.subheader("Filters - Filter Changes are updated dynamically in all the plots below")
    filter_col1, filter_col2 = st.columns(2)

    # Date filters: Start Date and End Date
    with filter_col1:
        start_date = st.date_input(
            "Start Date", 
            value=data['Purchase_Date'].min(), 
            help="Select the start date for filtering purchase data."
        )
        end_date = st.date_input(
            "End Date", 
            value=data['Purchase_Date'].max(), 
            help="Select the end date for filtering purchase data."
        )

    # Category filter: Dropdown to select a specific category
    with filter_col2:
        selected_category = st.selectbox(
            "Filter by Category", 
            options=[None] + list(data['Category'].unique()), 
            format_func=lambda x: "All" if x is None else x,
            help="Select a category to filter data. 'All' includes all categories."
        )

    # Filtering data based on user inputs
    filtered_data = data[
        (data['Purchase_Date'] >= pd.Timestamp(start_date)) & 
        (data['Purchase_Date'] <= pd.Timestamp(end_date))
    ]
    if selected_category:
        filtered_data = filtered_data[filtered_data['Category'] == selected_category]

    # Key Metrics Section: Displays important metrics based on filtered data
    st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)
    metrics = summary_metrics(filtered_data)

    # Display metrics in columns
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    with metric_col1:
        st.markdown(
            f"<div class='metric-card'><b>Total Revenue</b><br>{metrics['Total Revenue']}</div>",
            unsafe_allow_html=True
        )
    with metric_col2:
        st.markdown(
            f"<div class='metric-card'><b>Avg Revenue Per Unit</b><br>{metrics['Avg Revenue Per Unit']}</div>",
            unsafe_allow_html=True
        )
    with metric_col3:
        st.markdown(
            f"<div class='metric-card'><b>Top Category</b><br>{metrics['Top Category']}</div>",
            unsafe_allow_html=True
        )

    metric_col4, metric_col5 = st.columns(2)
    with metric_col4:
        st.markdown(
            f"<div class='metric-card'><b>Best Selling Product</b><br>{metrics['Best Selling Product']}</div>",
            unsafe_allow_html=True
        )
    with metric_col5:
        st.markdown(
            f"<div class='metric-card'><b>Most Profitable Product</b><br>{metrics['Most Profitable Product']}</div>",
            unsafe_allow_html=True
        )

    # Product Profitability Analysis: Visualizes product-level profitability
    st.markdown('<div class="section-title">Product Profitability Analysis (Hover over data points for more details)</div>', unsafe_allow_html=True)
    product_fig, product_summary = product_profitability_analysis(filtered_data, selected_category)
    st.plotly_chart(product_fig, use_container_width=True)
    st.markdown(f'<div class="summary-box">{product_summary}</div>', unsafe_allow_html=True)

    # Category Sales Analysis: Displays sales distribution across categories
    st.markdown('<div class="section-title">Category Sales Distribution</div>', unsafe_allow_html=True)
    category_fig, category_summary = category_sales_analysis(filtered_data)
    st.plotly_chart(category_fig, use_container_width=True)
    st.markdown(f'<div class="summary-box">{category_summary}</div>', unsafe_allow_html=True)

    # Subcategory Sales Analysis: Insights into subcategory performance
    st.markdown('<div class="section-title">Subcategory Sales Analysis</div>', unsafe_allow_html=True)
    subcategory_fig, subcategory_summary = subcategory_analysis(filtered_data)
    st.plotly_chart(subcategory_fig, use_container_width=True)
    st.markdown(f'<div class="summary-box">{subcategory_summary}</div>', unsafe_allow_html=True)

    # Customer-Level Analysis: Analysis at an individual customer level
    if 'Customer_ID' in filtered_data.columns:
        st.markdown('<div class="section-title">Customer-Level Analysis</div>', unsafe_allow_html=True)
        customer_ids = sorted(filtered_data['Customer_ID'].unique())

        # Dropdown to select a specific customer ID
        selected_customer_id = st.selectbox(
            "Select Customer ID", 
            options=customer_ids, 
            format_func=lambda x: f"Customer {x}",
            help="Select a customer ID for detailed analysis."
        )

        # If a customer is selected, display their specific analysis
        if selected_customer_id:
            customer_fig, customer_summary = user_analysis(filtered_data, selected_customer_id)
            st.plotly_chart(customer_fig, use_container_width=True)
            st.markdown(f'<div class="summary-box">{customer_summary}</div>', unsafe_allow_html=True)

def segmentation_page(data):
    """
    Renders the Customer Segmentation Analysis page.

    Parameters:
    - data (pd.DataFrame): The dataset used for segmentation analysis.

    The page includes:
    1. A header with a gradient background for visual appeal.
    2. A customer distribution chart to visualize segmentation.
    3. A detailed segment analysis with custom icons and colors for each segment.
    """

    # Header Section: Displays the title and a brief description
    st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2); 
                    color: white; padding: 25px; border-radius: 15px; 
                    margin-bottom: 30px; text-align: center;">
            <h1 style="margin: 0;">Customer Segmentation Analysis</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">Understanding Your Customer Groups</p>
        </div>
    """, unsafe_allow_html=True)

    # Segmentation Plot Section: Displays the customer distribution visualization
    segmentation_fig, segment_summaries = customer_segmentation(data)  # Generates the figure and segment summaries
    st.markdown("""
        <div style="background: white; padding: 20px; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px;">
            <h2 style="color: #4a5568; margin-bottom: 20px;">Customer Distribution - Hover over data points for added details</h2>
    """, unsafe_allow_html=True)
    st.plotly_chart(segmentation_fig, use_container_width=True)  # Embeds the plotly chart in the dashboard
    st.markdown("</div>", unsafe_allow_html=True)

    # Segment Summaries Section: Provides analysis for each customer segment
    st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h2 style="color: #4a5568; margin-bottom: 25px;">Segment Analysis</h2>
    """, unsafe_allow_html=True)

    # Icon and color definitions for each segment
    icons = {
        "Value Seekers": {"icon": "💰", "color": "#3182ce"},
        "Frequent Shoppers": {"icon": "🛍️", "color": "#805ad5"},
        "Occasional Buyers": {"icon": "🕒", "color": "#38a169"},
        "High Spenders": {"icon": "💎", "color": "#e53e3e"}
    }

    # Iterates through each segment and displays its analysis
    for segment, summary in segment_summaries.items():
        icon_data = icons.get(segment, {"icon": "📊", "color": "#718096"})  # Default icon and color if segment not found
        st.markdown(f"""
            <div style="background: #f8fafc; padding: 20px; border-radius: 10px; 
                        margin-bottom: 15px; border-left: 5px solid {icon_data['color']};">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <div style="font-size: 2em; margin-right: 15px;">
                        {icon_data['icon']}
                    </div>
                    <div style="flex-grow: 1;">
                        <h3 style="color: #2d3748; margin: 0;">{segment}</h3>
                    </div>
                </div>
                <div style="color: #4a5568; line-height: 1.6; background: white; 
                            padding: 15px; border-radius: 8px;">
                    {summary}  <!-- Segment-specific summary details -->
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Closing div for segment summaries section
    st.markdown("</div>", unsafe_allow_html=True)

import streamlit as st

def recommender_system_page(data):
    """
    Renders the Smart Product Recommendations page.

    Parameters:
    - data (pd.DataFrame): The dataset used for training the recommendation model and generating recommendations.

    The page includes:
    1. Header Section: Displays the title of the page.
    2. Model Training: Trains and initializes the recommendation model.
    3. Customer Selection: Allows the user to input a Customer ID to generate recommendations.
    4. Recommendations Section: Displays top product recommendations with details and scores.
    """

    # Header Section: Title for the recommendations page
    st.title("Smart Product Recommendations")

    # Model Training Section: Trains the recommendation model
    with st.spinner("Training recommendation engine..."):  # Show a spinner while the model is being trained
        model, data = train_and_save_model(data)  # Train the recommendation model and update the data
    st.success("Recommendation engine ready!")  # Notify the user when training is complete

    # Customer Selection Section: Input for Customer ID
    user_id = st.number_input(
        "Enter Customer ID:", 
        min_value=1, 
        max_value=int(data['Customer_ID'].max()), 
        step=1, 
        help="Input a valid Customer ID to get personalized recommendations."
    )

    if user_id:
        # Generate recommendations for the selected Customer ID
        recommendations = get_recommendations_for_user(user_id, n=5, data=data, model=model)

        # Header for Recommendations Section
        st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                <h2 style="color: #0066cc; margin: 0;">Top Recommendations for Customer #{user_id}</h2>
            </div>
        """, unsafe_allow_html=True)

        # Display each recommendation
        for idx, rec in enumerate(recommendations, 1):
            col1, col2 = st.columns([3, 1])  # Split layout: 3:1 for details and score

            # Recommendation Details Section
            with col1:
                st.markdown(f"""
                    <div style="background-color: white; padding: 20px; border-radius: 10px; 
                              border-left: 5px solid #0066cc; margin-bottom: 15px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <h3 style="color: #0066cc; margin: 0 0 10px 0;">Recommendation #{idx}</h3>
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr>
                                <td style="padding: 8px; color: #666;">Product ID:</td>
                                <td style="padding: 8px; font-weight: bold;">{rec['Product_ID']}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; color: #666;">Category:</td>
                                <td style="padding: 8px; font-weight: bold;">{rec['Category']}</td>
                            </tr>
                            <tr>
                                <td style="padding: 8px; color: #666;">Subcategory:</td>
                                <td style="padding: 8px; font-weight: bold;">{rec['Subcategory']}</td>
                            </tr>
                        </table>
                    </div>
                """, unsafe_allow_html=True)

            # Recommendation Score Section
            with col2:
                st.markdown(f"""
                    <div style="background-color: #e6f3ff; padding: 15px; border-radius: 10px; 
                              text-align: center; margin-bottom: 15px;">
                        <div style="color: #0066cc; font-size: 0.9em;">Score</div>
                        <div style="font-size: 1.5em; font-weight: bold; color: #0066cc;">
                            {rec['Estimated_Rating']:.1f}
                        </div>
                        <div style="color: #666; font-size: 0.8em;">out of 10.0</div>
                    </div>
                """, unsafe_allow_html=True)

            # Expandable Section: Explains the reason behind the recommendation
            with st.expander("Why this recommendation?"):
                st.markdown(f"""
                    <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px; color: #444;">
                        {rec['Reason']}  <!-- Explanation for the recommendation -->
                    </div>
                """, unsafe_allow_html=True)

            # Spacing between recommendations
            st.markdown("<br>", unsafe_allow_html=True)


def generate_report(data):
    """
    Generates the Business Intelligence Report page.

    Parameters:
    - data (pd.DataFrame): The dataset used for generating metrics, segmentation, and recommendations.

    The report includes:
    1. Insights from Data Analysis
    2. Customer Segmentation
    3. Example Recommendations
    """

    # Header Section: Displays the report title with a visually appealing gradient background
    st.markdown("""
        <div style="background: linear-gradient(135deg, #E3F2FD, #BBDEFB); 
                    color: #1a237e; padding: 25px; border-radius: 15px; 
                    margin-bottom: 30px; text-align: center;">
            <h1 style="margin: 0;">📄 Business Intelligence Report</h1>
        </div>
    """, unsafe_allow_html=True)

    # Insights from Data Analysis Section
    st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px;">
            <h2 style="color: #1a237e; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5em;">📊</span>
                <span>1. Insights from Data Analysis</span>
            </h2>
    """, unsafe_allow_html=True)

    metrics = summary_metrics(data)  # Generate summary metrics from the dataset
    st.markdown("""
        <div style="display: flex; flex-wrap: wrap; gap: 15px;">
    """, unsafe_allow_html=True)

    # Display metrics in cards with gradient backgrounds
    for key, value in metrics.items():
        st.markdown(f"""
            <div style="flex: 1 1 calc(33% - 10px); background: linear-gradient(135deg, #74ebd5, #9face6); 
                        color: white; padding: 20px; border-radius: 10px; text-align: center; 
                        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
                <div style="font-size: 1.2em; font-weight: bold;">{key}</div>
                <div style="font-size: 1.5em; margin-top: 5px;">{value}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)

    # Customer Segments Section
    st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin-bottom: 30px;">
            <h2 style="color: #1a237e; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5em;">👥</span>
                <span>2. Customer Segments</span>
            </h2>
    """, unsafe_allow_html=True)

    _, segment_summaries = customer_segmentation(data)  # Obtain segmentation data

    # Style definitions for each segment
    segment_styles = {
        "Value Seekers": {"icon": "💰", "color": "#3498db", "bg": "#ebf5fb"},
        "Frequent Shoppers": {"icon": "🛍️", "color": "#e74c3c", "bg": "#fce5e5"},
        "Occasional Buyers": {"icon": "🕒", "color": "#2ecc71", "bg": "#e8f8f5"},
        "High Spenders": {"icon": "💎", "color": "#9b59b6", "bg": "#f4ecf7"}
    }

    # Display customer segment summaries
    for segment, summary in segment_summaries.items():
        style = segment_styles.get(segment, {"icon": "📊", "color": "#34495e", "bg": "#f7f9fa"})
        st.markdown(f"""
            <div style="background: {style['bg']}; border-left: 4px solid {style['color']}; 
                        padding: 20px; border-radius: 10px; margin-bottom: 20px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
                <div style="display: flex; align-items: center; margin-bottom: 15px;">
                    <span style="font-size: 2em; margin-right: 15px;">{style['icon']}</span>
                    <h3 style="color: {style['color']}; margin: 0;">{segment}</h3>
                </div>
                <div style="color: #2c3e50; line-height: 1.6;">{summary}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Example Recommendations Section
    st.markdown("""
        <div style="background: white; padding: 25px; border-radius: 15px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);">
            <h2 style="color: #1a237e; margin-bottom: 20px; display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.5em;">🎯</span>
                <span>3. Example Recommendations</span>
            </h2>
    """, unsafe_allow_html=True)

    with st.spinner("Training recommender system model..."):  # Indicate model training process
        model, _ = train_and_save_model(data)  # Train recommendation model

    # Input for selecting Customer ID
    user_id = st.number_input("Select Customer ID for Example Recommendations:", 
                             min_value=1, max_value=int(data['Customer_ID'].max()), step=1,
                             help="Input a Customer ID to view personalized recommendations.")

    if user_id:
        # Display recommendations for the selected Customer ID
        st.markdown(f"""
            <div style="margin: 20px 0; padding: 15px; background: #f8f9fa; 
                        border-radius: 10px; border-left: 4px solid #1a237e;">
                <div style="font-size: 1.3em; color: #1a237e; font-weight: bold;">
                    Recommendations for Customer #{user_id}
                </div>
            </div>
        """, unsafe_allow_html=True)

        recommendations = get_recommendations_for_user(user_id, n=3, data=data, model=model)

        # Display recommendations in cards
        st.markdown("<div style='display: flex; flex-wrap: wrap; gap: 15px;'>", unsafe_allow_html=True)
        for rec in recommendations:
            st.markdown(f"""
                <div style="flex: 1 1 calc(33% - 10px); background: white; 
                            border: 1px solid #e0e0e0; padding: 20px; border-radius: 10px; 
                            box-shadow: 0 2px 8px rgba(0,0,0,0.1);">
                    <div style="font-size: 1.1em; color: #1a237e; font-weight: bold; margin-bottom: 10px;">
                        Product ID: {rec['Product_ID']}
                    </div>
                    <div style="background: #f8f9fa; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
                        <div style="color: #1a237e; font-weight: bold;">Rating: {rec['Estimated_Rating']:.2f}/10.0</div>
                    </div>
                    <div style="margin: 5px 0; font-size: 0.95em;">
                        <b>Category:</b> {rec['Category']}
                    </div>
                    <div style="margin: 5px 0; font-size: 0.95em;">
                        <b>Subcategory:</b> {rec['Subcategory']}
                    </div>
                    <div style="margin-top: 15px; padding: 10px; background: #f8f9fa; 
                                border-radius: 8px; font-size: 0.9em;">
                        <b>Why this recommendation?</b><br>
                        {rec['Reason']}
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


tabs = st.tabs(["Analysis", "Customer Segmentation", "Recommender System", "Report"])
with tabs[0]:
    analysis_page(data)

with tabs[1]:
    segmentation_page(data)

with tabs[2]:
    recommender_system_page(data)

with tabs[3]:
    generate_report(data)
