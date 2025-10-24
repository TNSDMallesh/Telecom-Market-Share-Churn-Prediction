<h1 align="center" style="font-size:3rem; font-weight:bold;">
📊 Telecom Market Share & Churn Prediction
</h1>

---

<div align="center">

## 📘 Overview

This project focuses on predicting telecom market churn and analyzing operator-level market share trends using multi-year time-series data (2009–2025). The goal is to identify operator-month segments at risk of subscriber decline, enabling proactive retention strategies and improved market positioning.

Through comprehensive data preprocessing, feature engineering, and machine learning modeling, this project provides a scalable solution for telecom operators to anticipate customer churn and optimize business decisions with data-backed insights.

</div>

---

## 🧠 Key Objectives

- Predict customer churn and analyze market competition.
- Engineer advanced features like velocity-of-change metrics and Herfindahl–Hirschman Index (HHI) to capture market dynamics.
- Compare multiple ML & DL models to identify the most accurate churn prediction approach.
- Visualize telecom trends and churn behavior through detailed EDA.

## 🏗️ Project Structure


Telecom-Market-Share-Churn-Prediction/
│
├── data/ # Raw and processed telecom datasets
├── notebooks/ # Jupyter notebooks for analysis & modeling
├── src/ # Source code for preprocessing, modeling, utils
│ ├── data_preprocessing.py
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── visualization.py
│ └── evaluation.py
├── results/ # Model outputs, metrics, and plots
├── requirements.txt # Required dependencies
├── README.md # Project documentation
└── main.py # Entry point for running the full pipeline
text

## ⚙️ Installation Steps


git clone https://github.com/TNSDMallesh/Telecom-Market-Share-Churn-Prediction.git
cd Telecom-Market-Share-Churn-Prediction
Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate # Linux/Mac
venv\Scripts\activate # Windows
Install dependencies
pip install -r requirements.txt
text

## 🧩 Usage

- To run the full prediction pipeline:  
  `python main.py`

- To explore data analysis or experiment with models:  
  `jupyter notebook notebooks/`

- Modify parameters (model type, hyperparameters, time period) in the `config.py` file for customized experimentation.

## 🧰 Technologies Used

- Python libraries: Pandas, NumPy, Scikit-learn, XGBoost, TensorFlow/Keras  
- Visualization: Matplotlib, Seaborn  
- Data Handling: CSV, Pickle, Joblib  
- Development: Jupyter Notebooks, Python Scripts

## 📈 Results

After rigorous model evaluation, the Random Forest algorithm achieved the best predictive performance with an F1-score of approximately 0.9979, outperforming other models such as Logistic Regression, Gradient Boosting, XGBoost, and LSTM.

The project also visualizes trends in subscriber churn, market share variations, and competitive positioning across operators and years.

## 🌐 Future Enhancements

- Integrate real-time data streams for dynamic churn updates.  
- Deploy model through Flask or FastAPI for business applications.  
- Incorporate reinforcement learning for optimized retention policy simulations.

## 🤝 Contributing

Contributions, suggestions, and issue reports are welcome! Fork the repository, make improvements, and submit a pull request.

## 📄 License

This project is open-source under the MIT License.
