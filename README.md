
# Smart Retail Store Dashboard 
## The project has been deployed on the attached link: [Deployed Project](https://retailstore.streamlit.app)

The Retail Store Customer Dashboard provides an intuitive interface for analyzing customer purchase behavior. It features data visualization, segmentation, and personalized recommendations to optimize decision-making. This tool empowers businesses to improve customer retention and enhance sales strategies.

---



## ⚙️ Project Setup

### 1. Clone the Repository
To start working with the project, first, clone the repository using the following command:

```bash
git clone https://github.com/Devil1402/retailStoreHomework.git
```

Navigate into the project directory:

```bash
cd retailStoreHomework
```

### 2. Create a Virtual Environment (venv)
Set up a virtual environment to manage dependencies:

```bash
python3 -m venv venv
```

### 3. Activate the Virtual Environment
Activate the virtual environment using the appropriate command for your operating system:

For macOS/Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
.\venv\Scripts\activate
```

### 3.Project Directory
The contents are placed correctly by defualt. If not please place them in the root directory accordingly

```bash
retailHomeWork/
├── analysis.py
├── app.py
├── best_params.pkl
├── config.py
├── customerPurchaseData.csv
├── dataGeneration.py
├── model.pkl
├── recommender_system.py
├── requirements.txt
├── segmentation.py
```

### 4. Install Required Dependencies
Once the virtual environment is activated, install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

### 5. Steps before running the streamlit command
1. The dataset has already been given in the dataset and the path is stored in the config.py file.
2. If you want to try running the synthetic data generator to test the data generation, in your terminal, type in the command
```bash
python dataGeneration.py
```
3. The customer segmentation and recommender system models are already trained. Training the recommender system model again requires a lot of time, so the pickle files have already been provided which can be directly used to get recommendations


### 5. Running the Streamlit App
To run the Streamlit app, execute the following command from the terminal:

```bash
streamlit run app.py
```

This will start the Retail Store Dashboard web interface, where you can interact with the system to get customer insights.

---

## 💡 Additional Notes
- Ensure that all dependencies are properly installed in the virtual environment.
- Data storage guidelines must be followed to ensure no errors.
- The app can be customized further based on specific use cases and requirements.

---

🌟 We hope this guide helps you set up and run the Contract Query ChatBot with ease! If you have any issues, feel free to create an issue on the [GitHub repository](https://github.com/Devil1402/retailStoreHomework/issues).

---

## Screenshots

1. Dynamic Filtering
<img width="1469" alt="image" src="https://github.com/user-attachments/assets/17eb03f8-ba58-41a8-8b24-79d2508aa55a" />

2. Interractive plots
<img width="1469" alt="image" src="https://github.com/user-attachments/assets/df926e12-a984-4ea3-b1e1-e643470f98a6" />

3. Interractive Customer Segmentation Analysis
<img width="1469" alt="image" src="https://github.com/user-attachments/assets/c298c792-23bb-4676-9979-5f5bd8da9500" />

4. Robust Recommendation System
<img width="1469" alt="image" src="https://github.com/user-attachments/assets/737180ea-00e2-4586-8630-8eafce9bed81" />

5. Intuitive Business Reports
<img width="1469" alt="image" src="https://github.com/user-attachments/assets/1716d7f7-84f8-462a-a792-a28633903c1b" />















