
# Smart Retail Store Dashboard 
## [Deployed Project](https://retailstore.streamlit.app)

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
