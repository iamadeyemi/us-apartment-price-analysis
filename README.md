
---

# **🏡 Unlocking Rental Market Trends**  
🔍 **Analyzing Apartment Prices and Features Across the U.S.**  

![Rental Market Analysis](https://source.unsplash.com/1600x900/?real-estate,apartment)  

## 📌 **About the Project**  
The U.S. rental market is constantly evolving due to economic shifts, demographic changes, and housing demand. This project leverages **data science and machine learning** to analyze key pricing factors, helping renters, landlords, and policymakers make informed decisions.  

This study explores how **location, square footage, number of bedrooms, and amenities** impact rental prices. Using **Multiple Linear Regression and Binomial Logistic Regression**, the project predicts rental prices and classifies properties into high- and low-price categories. The final model is deployed as a **Flask-based web app** for real-time rental price predictions.  

---

## 🚀 **Key Features**  
✅ **📊 Data Exploration & Cleaning:** Handles missing data, removes duplicates, and preprocesses features for analysis.  
✅ **📈 Exploratory Data Analysis (EDA):** Identifies trends, correlations, and outliers in rental pricing.  
✅ **🧠 Machine Learning Models:** Trains and evaluates **Linear Regression, Polynomial Regression, Random Forest, and Support Vector Regression (SVR)** models.  
✅ **💰 Price Prediction Model:** Uses regression techniques to estimate rental prices.  
✅ **🏡 Price Classification:** Categorizes properties into **high-priced vs. low-priced** using logistic regression.  
✅ **🌎 Interactive Data Visualizations:** Heatmaps, scatter plots, and bar charts to uncover key insights.  
✅ **🔥 Flask Deployment:** Deploys the trained model in a web app for real-time rental price predictions.  

---

## 📊 **Data Overview**  

- **Dataset Source:** UCI Machine Learning Repository  
- **Observations:** 10,000+ apartment listings  
- **Features Include:**  
  - 🏙 **Location Data:** State, city, latitude, longitude  
  - 🏠 **Apartment Features:** Number of bedrooms, bathrooms, square footage  
  - 💲 **Rental Price:** Target variable  
  - 🏢 **Property Type:** High-rise, townhouse, detached home  
  - 🎓 **Demographics:** Neighborhood population, median income  

---

## 🛠 **Technologies Used**  

### **📌 Programming & Libraries**  
- **Python 3.11.4** 🐍  
- **Scikit-Learn** (Machine Learning)  
- **Pandas & NumPy** (Data Manipulation)  
- **Matplotlib & Plotly** (Data Visualization)  
- **Flask** (Web Deployment)  

### **📌 Model Selection & Evaluation**  
- **Multiple Linear Regression** 📈  
- **Polynomial Regression (Degree 2)** 🔄  
- **Random Forest Regressor** 🌳  
- **Support Vector Regression (SVR)** 📊  
- **Evaluation Metrics:**  
  - Mean Absolute Error (MAE)  
  - Mean Squared Error (MSE)  
  - Root Mean Squared Error (RMSE)  
  - R² Score  

---

## 🔥 **Model Training & Testing**  

### **📌 Steps Followed**  
1️⃣ **Data Preprocessing:** Clean missing values, handle categorical data, remove duplicates.  
2️⃣ **Exploratory Data Analysis (EDA):** Identify trends, price distributions, and correlations.  
3️⃣ **Feature Engineering:** Create new meaningful features such as `Study_Tutoring_Interaction`.  
4️⃣ **Train-Test Split:** 80% for training, 20% for testing to ensure model generalization.  
5️⃣ **Model Training:** Fit multiple ML models and compare performance metrics.  
6️⃣ **Hyperparameter Tuning:** Optimize models for best performance.  
7️⃣ **Model Evaluation:** Use regression metrics to assess accuracy.  
8️⃣ **Deployment:** Deploy the best model using Flask for real-time predictions.  

---

## 🎯 **Results & Insights**  

📌 **Key Findings from EDA:**  
- **Rental prices are strongly correlated** with apartment size (square footage) and the number of bathrooms.  
- **High-rent properties** are mostly found in major metropolitan areas.  
- **The number of bedrooms has a negative correlation** with price when controlled for square footage.  

📌 **Best Performing Model:**  
🏆 **Random Forest Regressor** achieved the highest accuracy in predicting rental prices.  

---

## 💻 **Web App Deployment**  

### **🏡 How to Use the Flask App**  
1️⃣ **Clone the repository**  
```bash
git clone https://github.com/yourusername/rental-market-trends.git
cd rental-market-trends
```
2️⃣ **Install dependencies**  
```bash
pip install -r requirements.txt
```
3️⃣ **Run the Flask app**  
```bash
python app.py
```
4️⃣ **Access the web app** at `http://127.0.0.1:5000`  

🔹 **Enter apartment details** (square footage, bedrooms, location)  
🔹 **Click Predict** to get estimated rental price  
🔹 **View insights from trained ML models**  

---

## 📌 **Future Improvements**  
🔹 Integrate deep learning models for enhanced accuracy.  
🔹 Add more features like crime rate, school proximity, and economic indicators.  
🔹 Improve Flask app UI for a better user experience.  
🔹 Deploy the app on a cloud platform like **Heroku or AWS**.  

---

## 👨‍💻 **Contributors**  
- **Timothy Adeyemi** 🚀  
  - **GitHub:** [@yourusername](https://github.com/iamtimothy)  
  - **LinkedIn:** [Your Profile](https://www.linkedin.com/in/timothy-ade/)  

---

## 📜 **License**  
This project is licensed under the **MIT License** – feel free to use and improve it!  

---

## ⭐ **Show Your Support!**  
If you found this project useful, please **star ⭐ the repository** and share it!  

Happy coding! 🚀🏡💻  