import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVDpp
from surprise.model_selection import GridSearchCV, train_test_split
import pickle
import os

def train_and_save_model(data):
    """
    Train the recommender model or load the pre-trained model if available.

    Args:
        data (pd.DataFrame): Input dataset containing customer and product purchase data.

    Returns:
        tuple: A trained model and the processed dataset.
    """
    # Scale purchase amounts for model compatibility
    data['Purchase_Amount_Scaled'] = np.log1p(data['Purchase_Amount'])

    # Prepare data for Surprise library
    reader = Reader(rating_scale=(data['Purchase_Amount_Scaled'].min(), data['Purchase_Amount_Scaled'].max()))
    data_surprise = Dataset.load_from_df(data[['Customer_ID', 'Product_ID', 'Purchase_Amount_Scaled']], reader)

    # Check for existing model files
    if os.path.exists('model.pkl') and os.path.exists('best_params.pkl'):
        # Load pre-trained model and parameters
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('best_params.pkl', 'rb') as f:
            best_params = pickle.load(f)
    else:
        # Define grid search parameters
        param_grid = {
            "n_factors": [20, 50, 100],
            "n_epochs": [10, 20, 30],
            "lr_all": [0.002, 0.005, 0.01],
            "reg_all": [0.02, 0.05, 0.1],
        }

        # Perform grid search for optimal parameters
        grid_search = GridSearchCV(SVDpp, param_grid, measures=["rmse", "mae"], cv=3, n_jobs=-1)
        grid_search.fit(data_surprise)

        # Train the model with the best parameters
        best_params = grid_search.best_params["rmse"]
        model = SVDpp(**best_params)
        trainset, testset = train_test_split(data_surprise, test_size=0.2)
        model.fit(trainset)

        # Save the model and parameters
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        with open('best_params.pkl', 'wb') as f:
            pickle.dump(best_params, f)

    return model, data

def get_recommendations_for_user(user_id, n=5, data=None, model=None):
    """
    Generate product recommendations for a specific user using a pre-trained model.

    Args:
        user_id (int): The ID of the user for whom recommendations are generated.
        n (int): Number of recommendations to generate. Defaults to 5.
        data (pd.DataFrame): Input dataset containing customer and product purchase data.
        model: Trained recommender model.

    Returns:
        list: A list of dictionaries containing recommended products with details and reasoning.
    """
    all_items = data['Product_ID'].unique()
    user_items = data[data['Customer_ID'] == user_id]['Product_ID'].unique()
    items_to_predict = [item for item in all_items if item not in user_items]

    # Predict ratings for items not yet purchased by the user
    predictions = [
        (item, model.predict(uid=user_id, iid=item).est)
        for item in items_to_predict
    ]

    # Sort predictions by estimated rating
    predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:n]

    # Retrieve product details
    product_details = data[['Product_ID', 'Category', 'Subcategory']].drop_duplicates()
    product_details_dict = product_details.set_index('Product_ID').T.to_dict()

    detailed_recommendations = [
        {
            "Product_ID": iid,
            "Estimated_Rating": est,
            "Category": product_details_dict[iid]["Category"],
            "Subcategory": product_details_dict[iid]["Subcategory"]
        }
        for iid, est in predictions
    ]

    # Add reasoning for recommendations
    for recommendation in detailed_recommendations:
        category_products = data[data['Category'] == recommendation['Category']]
        subcategory_products = category_products[category_products['Subcategory'] == recommendation['Subcategory']]

        user_past_purchases = data[data['Customer_ID'] == user_id]
        user_category_purchases = user_past_purchases[user_past_purchases['Category'] == recommendation['Category']]
        user_subcategory_purchases = user_category_purchases[user_category_purchases['Subcategory'] == recommendation['Subcategory']]

        # Generate reasoning
        reason_parts = []
        if not user_category_purchases.empty:
            reason_parts.append(f"the customer has shown a strong interest in {recommendation['Category']} items previously")
        if not user_subcategory_purchases.empty:
            reason_parts.append(f"specifically within the {recommendation['Subcategory']} subcategory")

        popularity_score = subcategory_products['Purchase_Amount'].sum()
        reason_parts.append(f"this product is highly popular among other customers with total revenue of ${popularity_score:,.2f}")

        reason_parts.append(f"the predicted rating for this product is {recommendation['Estimated_Rating']:.2f}, indicating high satisfaction potential")

        recommendation['Reason'] = f"Recommended because {'; '.join(reason_parts)}."

    return detailed_recommendations

def analyze_popularity(data):
    """
    Analyze product popularity based on purchase counts and total revenue.

    Args:
        data (pd.DataFrame): Input dataset containing product purchase details.

    Returns:
        pd.DataFrame: A dataframe containing aggregated popularity metrics for products.
    """
    product_popularity = data.groupby('Product_ID').agg(
        total_purchases=('Purchase_Quantity', 'sum'),
        total_revenue=('Purchase_Amount', 'sum'),
        average_price=('Unit_Price', 'mean')
    ).sort_values(by='total_revenue', ascending=False)

    return product_popularity