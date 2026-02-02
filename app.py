from flask import Flask, request, render_template
import numpy as np
from classifier import load_model

app = Flask(__name__)
model = load_model()  # loads FIRST

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    
    # Extract numeric values safely
    showroom_price = float(data['showroom_price'])
    kms_driven = float(data['kms_driven'])
    owners = float(data['owners'])
    year = int(data['year'])
    
    # Encode categorical features to match training data (0/1/2 numeric values)
    fuel_map = {'Petrol': 0, 'Diesel': 1, 'CNG': 2}
    seller_map = {'Dealer': 0, 'Individual': 1}
    transmission_map = {'Manual': 0, 'Automatic': 1}
    
    fuel = fuel_map[data['fuel']]
    seller = seller_map[data['seller_type']]
    transmission = transmission_map[data['transmission']]
    
    # Calculate car age (since training data likely used age instead of year)
    age = 2026 - year  # Current year is 2026
    
    # Create 8 features matching your train_reg.py X array structure
    # Order: [showroom_price, kms_driven, owners, age, fuel, seller, transmission, constant_feature]
    features = [
        showroom_price,    # 0
        kms_driven,        # 1  
        owners,            # 2
        age,               # 3 (calculated from year)
        fuel,              # 4
        seller,            # 5
        transmission,      # 6
        1.0                # 7 (constant feature matching training sample)
    ]
    
    final_features = np.array([features])
    prediction = model.predict(final_features)
    
    return render_template(
        "index.html",
        prediction_text=f"Predicted Selling Price: ₹{prediction[0]:.2f} Lakhs"
    )

if __name__ == "__main__":
    app.run(debug=True)
