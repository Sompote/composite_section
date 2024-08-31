import streamlit as st
import torch
import joblib
import numpy as np

# Define the MLP model
class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, hidden_size3, output_size):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size1)
        self.fc2 = torch.nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = torch.nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = torch.nn.Linear(hidden_size3, output_size)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Load the model and scalers
@st.cache_resource
def load_model_and_scalers():
    model = MLP(input_size=12, hidden_size1=36, hidden_size2=72, hidden_size3=36, output_size=1)
    model.load_state_dict(torch.load('mlp_model.pth'))
    model.eval()
    scaler_X = joblib.load('scaler_X.joblib')
    scaler_Y = joblib.load('scaler_Y.joblib')
    return model, scaler_X, scaler_Y

# Load the model and scalers
model, scaler_X, scaler_Y = load_model_and_scalers()

st.title('Composite Column Load Prediction')

# Create input fields for each feature with detailed descriptions and initial values
fc = st.number_input("Concrete strength (MPa)", min_value=0.0, value=43.2)
H = st.number_input("Height of column (mm)", min_value=0.0, value=477.0)
Do = st.number_input("Diameter of outer steel tube (from outer to outer) (mm)", min_value=0.0, value=159.0)
Di = st.number_input("Diameter of outer steel tube (from inner to inner) (mm)", min_value=0.0, value=156.2)
fy_ost = st.number_input("Yield strength of outer steel tube (MPa)", min_value=0.0, value=290.0)
fy_rebar = st.number_input("Yield strength of longitudinal reinforcing rebars (MPa)", min_value=0.0, value=0.0)
rho_l = st.number_input("Ratio of longitudinal reinforcing rebars", min_value=0.0, max_value=1.0, value=0.0)
rho_v = st.number_input("Volumetric ratio of circular/spiral reinforcing rebars", min_value=0.0, max_value=1.0, value=0.0)
fy_spiral = st.number_input("Yield strength of circular/spiral reinforcing rebars (MPa)", min_value=0.0, value=0.0)
As_iss = st.number_input("Total area of internal reinforcing steel section (mm²)", min_value=0.0, value=754.16)
fy_iss = st.number_input("Yield strength of internal reinforcing steel section (MPa)", min_value=0.0, value=289.834)
e = st.number_input("Eccentricity (mm)", min_value=0.0, value=0.0)

if st.button('Predict Load Capacity'):
    # Prepare input data
    input_data = np.array([[fc, H, Do, Di, fy_ost, fy_rebar, rho_l, rho_v, fy_spiral, As_iss, fy_iss, e]])
    
    # Normalize input data
    input_normalized = scaler_X.transform(input_data)
    
    # Convert to PyTorch tensor
    input_tensor = torch.tensor(input_normalized, dtype=torch.float32)
    
    # Make prediction
    with torch.no_grad():
        prediction_normalized = model(input_tensor)
    
    # Denormalize prediction
    prediction = scaler_Y.inverse_transform(prediction_normalized.numpy())
    
    st.success(f'Predicted Load Capacity: {prediction[0][0]:.2f} kN')

st.write("Note: This app uses a pre-trained MLP model to predict the load capacity of a composite column based on the input parameters.")
