import numpy as np
from panel_solver import panel_method

def test_cylinder():
    print("--- Running Cylinder Test (Kutta = False) ---")
    
    # Generate perfect circle with Clockwise winding starting from x=1, y=0
    # Negative angle sweep creates the CW direction
    theta_nodes = np.linspace(0, -2*np.pi, 100)
    X = np.cos(theta_nodes)
    Y = np.sin(theta_nodes)
    
    # Solve pure source panel method (kutta=False means no vortex)
    Cp_calc, Cl, Cd = panel_method(X, Y, alpha_deg=0, kutta=False)
    
    # Calculate analytical Cp for a cylinder
    # We evaluate this at the control points (the physical center of each panel)
    theta_c = (theta_nodes[:-1] + theta_nodes[1:]) / 2
    Cp_analytic = 1.0 - 4.0 * np.sin(theta_c)**2
    
    # Find the biggest difference between our code's answer and the perfect math answer
    max_error = np.max(np.abs(Cp_calc - Cp_analytic))
    
    print(f"Max Cp Error vs Analytic: {max_error:.2e}")
    print(f"Calculated Drag Coeff (Cd): {Cd:.2e}")
    
    if max_error < 1e-2:
        print("PASS: Source matrix equations are correct.")
    else:
        print("FAIL: Check influence coefficient math.")

if __name__ == "__main__":
    test_cylinder()