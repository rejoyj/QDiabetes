import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from qiskit_machine_learning.algorithms import VQC
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import SPSA 
from qiskit_aer import Aer

# 1. Power Run Data Loader
def prepare_diabetes_data(filename):
    base_path = os.path.dirname(__file__)
    filepath = os.path.join(base_path, filename)
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filename} at {filepath}.")

    data = pd.read_csv(filepath)
    
    # FIX: Select exactly 7 features to match the 7-qubit circuit
    # Indices: 0(Pregnancies), 1(Glucose), 2(BloodPressure), 4(Insulin), 5(BMI), 6(Pedigree), 7(Age)
    X = data.iloc[:, [0, 1, 2, 4, 5, 6, 7]].values 
    y = data['Outcome'].values

    # Scaling is essential for Quantum Feature Maps
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # One-hot encoding labels (Required for VQC)
    encoder = OneHotEncoder(sparse_output=False)
    y_one_hot = encoder.fit_transform(y.reshape(-1, 1))

    return train_test_split(X_scaled, y_one_hot, test_size=0.2, random_state=42)

print("--- Step 1: Loading 7 High-Impact Features ---")
train_X, test_X, train_y, test_y = prepare_diabetes_data('diabetes.csv')

# 2. Design the "Power Run" Quantum Brain
num_qubits = 7
# Added 'linear' entanglement to help qubits share information
feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=1, entanglement='linear')
ansatz = RealAmplitudes(num_qubits=num_qubits, reps=1)

# 3. Setup SPSA Optimizer
# Reduced maxiter to 150 to stop before the "flat" Barren Plateau
optimizer = SPSA(maxiter=150) 

def callback_graph(nfev, weights, obj_value, stepsize, accepted):
    if nfev % 30 == 0:
        print(f"Iteration {nfev}: Current Loss = {obj_value:.4f}")

# 4. Initialize and Train the VQC
vqc = VQC(
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph
)

print(f"\n--- Step 2: Training Power Run Model (7 Qubits, 150 MaxIter) ---")
print("This focuses on medical context and information sharing between qubits.")
vqc.fit(train_X, train_y)

# 5. Check Results
score = vqc.score(test_X, test_y)
accuracy_pct = score * 100
print("-" * 40)
print(f"Project Complete!")
print(f"Power Run Quantum Accuracy: {accuracy_pct:.2f}%")
print("-" * 40)

# 6. AUTO-REPORT GENERATION
with open("quantum_report.txt", "w") as f:
    f.write("QUANTUM DIABETES RISK PREDICTOR - PROJECT REPORT\n")
    f.write("================================================\n")
    f.write(f"Device: ThinkPad T580 (Linux Simulator)\n")
    f.write(f"Quantum Architecture: VQC with {num_qubits} Qubits\n")
    f.write(f"Optimization Strategy: Linear Entanglement & Early Stopping\n")
    f.write(f"Final Accuracy: {accuracy_pct:.2f}%\n")
    f.write("================================================\n")

print(f"\n[SUCCESS] Report updated in 'quantum_report.txt'")