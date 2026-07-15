import numpy as np
from panel_solver import naca4_coordinates, panel_method

def test_naca0012():
    print("--- Running Symmetric Airfoil Test (NACA 0012) ---")
    
    # Generate NACA 0012 geometry
    X, Y = naca4_coordinates('0012', n_panels=160)
    
    # Run at alpha = 0 to verify symmetry
    _, Cl_0, _ = panel_method(X, Y, alpha_deg=0.0, kutta=True)
    print(f"Cl at alpha=0: {Cl_0:.5f} (Expected: ~0.0)")
    
    # Sweep alpha to calculate the lift slope (dCl/dalpha)
    alphas = np.array([-2, 0, 2, 4])
    Cls = []
    
    for a in alphas:
        _, cl, _ = panel_method(X, Y, alpha_deg=a, kutta=True)
        Cls.append(cl)
        
    # Calculate slope in radians (using numpy polyfit to fit a straight line)
    alphas_rad = np.radians(alphas)
    slope, _ = np.polyfit(alphas_rad, Cls, 1)
    
    print(f"Cl-vs-alpha slope: {slope:.3f} per radian")
    print(f"Theoretical thin-airfoil slope: {2*np.pi:.3f} per radian")
    
    if 6.28 < slope < 7.0:
        print("PASS: Slope is just above 2*pi, validating thickness effects and the Kutta condition.")
    else:
        print("FAIL: Slope out of expected bounds.")

if __name__ == "__main__":
    test_naca0012()