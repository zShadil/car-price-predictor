import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, VotingRegressor

# Training data matching your exact form fields + app.py logic
# Features order: [showroom_price, kms_driven, owners, age, fuel, seller_type, transmission, constant]
X = np.array([
    # showroom_price(lakhs), kms_driven, owners, age(yrs), fuel(0=Petrol/1=Diesel/2=CNG), seller(0=Dealer/1=Individual), trans(0=Manual/1=Auto), constant
    [9.85, 6900, 0, 3, 0, 1, 0, 1],    # Petrol, Individual, Manual → Price: 7.5
    [5.50, 50000, 1, 6, 1, 0, 1, 1],   # Diesel, Dealer, Auto → Price: 4.2
    [7.20, 30000, 0, 4, 0, 1, 0, 0],   # Petrol, Individual, Manual → Price: 6.1
    [3.50, 70000, 2, 9, 1, 0, 1, 1],   # Diesel, Dealer, Auto → Price: 2.8
    [12.50, 25000, 0, 2, 0, 0, 1, 1],  # Petrol, Dealer, Auto → Price: 11.2
    [8.20, 45000, 1, 5, 2, 1, 0, 1],   # CNG, Individual, Manual → Price: 6.8
])

# Corresponding selling prices (in lakhs)
y = np.array([7.5, 4.2, 6.1, 2.8, 11.2, 6.8])

# Create and train VotingRegressor (same as before)
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=100, random_state=42)

model = VotingRegressor([
    ("lr", lr),
    ("rf", rf)
])

# Fit the model
model.fit(X, y)

# Save the trained model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ model.pkl created successfully!")
print(f"✅ Training completed with {len(X)} samples")
print(f"✅ Feature order: [showroom_price, kms_driven, owners, age, fuel, seller_type, transmission, constant]")
print("✅ Ready for Flask predictions!")
