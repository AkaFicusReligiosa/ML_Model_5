
# House Price Prediction using XGBRegressor

This project uses the **XGBoost Regression model (XGBRegressor)** to predict house prices based on various property features. The model leverages the power of gradient boosting to deliver accurate and robust predictions, making it suitable for real estate valuation and housing market analysis.

---

## 🎯 Objective

To develop a regression model using **XGBRegressor** that can predict the sale price of a house based on its features such as area, number of rooms, location, etc.

---

## 📁 Dataset

- **Source**: [Kaggle - House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
- **Target Variable**: `SalePrice`
- **Features**: Includes numerical and categorical attributes like:
  - LotArea
  - OverallQual
  - YearBuilt
  - TotalBsmtSF
  - GrLivArea
  - GarageCars
  - FullBath
  - Neighborhood (categorical)
  - And many more...

---

## 🚀 Technologies Used

- Python
- Pandas, NumPy
- XGBoost (XGBRegressor)
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 🧠 Model Workflow

1. **Load and Inspect Dataset**
2. **Data Preprocessing**
   - Handle missing values
   - Encode categorical variables
   - Feature scaling (if needed)
3. **Train/Test Split**
4. **Model Training**
   - Train `XGBRegressor` with hyperparameter tuning
5. **Model Evaluation**
   - Evaluate performance using RMSE, MAE, R² score
6. **Predictions on New Data**

---

## 📊 Evaluation Metrics

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

## 🧪 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/house-price-xgboost.git
   cd house-price-xgboost
````

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:

   ```bash
   jupyter notebook house_price_prediction.ipynb
   ```

---

## 🔮 Sample Prediction Code

```python
sample_input = [[8450, 7, 2003, 856, 1710, 2, 2]]  # Example feature values
predicted_price = model.predict(sample_input)
print("Predicted House Price: $", round(predicted_price[0], 2))
```

---

## 📈 Results

* XGBoost achieved a high R² score and low RMSE after hyperparameter tuning.
* Outperformed baseline linear regression in both accuracy and stability.

---

## 📜 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

* [Kaggle: House Prices Dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques)
* XGBoost documentation
* Scikit-learn
* Python open-source community

```


