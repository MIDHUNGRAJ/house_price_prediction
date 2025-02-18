import streamlit as st
import numpy as np
import your_dataset as ds
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# My Linear Regression Prediction
class MyLinearRegression:
    def fit(self, xdata, ydata):
        X_add_bias = np.c_[np.ones((xdata.shape[0], 1)), xdata]
        X_transpose = X_add_bias.T
        X_transpose_X = np.dot(X_transpose, X_add_bias)
        X_transpose_y = np.dot(X_transpose, ydata)
        self.theta = np.linalg.solve(X_transpose_X, X_transpose_y)

    def predict(self, x):
        X_with_bias = np.c_[np.ones((x.shape[0], 1)), x]
        return X_with_bias @ self.theta


# Load and preprocess data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(ds.X_train)
X_test_scaled = scaler.transform(ds.X_test)

my_reg = MyLinearRegression()
my_reg.fit(X_train_scaled, ds.y_train)

# Model Evaluation
y_pred = my_reg.predict(X_test_scaled)
r2 = r2_score(ds.y_test, y_pred)

# Streamlit App
st.title("House Price Prediction")
st.write("This app predicts house prices based on your inputs.")

u_area_in = st.slider(
    "Enter the Area (sq. ft.)", min_value=500, max_value=10000, step=100
)
u_be_rs_in = st.slider("Number of Bedrooms", min_value=1, max_value=6, step=1)
u_ba_rs_in = st.slider("Number of Bathrooms", min_value=1, max_value=6, step=1)
u_st_in = st.slider("Number of Stories", min_value=1, max_value=4, step=1)
u_pk_in = st.slider("Number of Parking Spots", min_value=0, max_value=3, step=1)

# Input Transformation
t_in = scaler.transform([[u_area_in, u_be_rs_in, u_ba_rs_in, u_st_in, u_pk_in]])

# Prediction
try:
    price = my_reg.predict(t_in)
    st.write("Predicted Price: ", round(price[0], 2))
except Exception as e:
    st.error(f"Error in prediction: {e}")

st.write(f"Model R2 Score: {r2:.2f}")
