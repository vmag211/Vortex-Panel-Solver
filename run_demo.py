import os
import numpy as np
import matplotlib.pyplot as plt
from panel_solver import naca4_coordinates, panel_method

def run_demo(code='2412'):
    print(f"--- Running Demo for NACA {code} ---")
    
    # Ensure the results directory exists so the code doesn't crash trying to save
    os.makedirs('results', exist_ok=True)
    
    # 1. Geometry Plot
    X, Y = naca4_coordinates(code, n_panels=160)
    plt.figure(figsize=(10, 3))
    plt.plot(X, Y, '-k', linewidth=1.5) # '-k' means a solid black line
    plt.title(f"NACA {code} Geometry")
    plt.axis('equal') # Forces the X and Y axes to have the same scale so the wing isn't stretched
    plt.grid(True, linestyle=':')
    plt.savefig(f"results/naca{code}_geometry.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved geometry plot.")
    
    # 2. Pressure Distribution (Cp) at Alpha = 4 degrees
    alpha = 4.0
    Cp, Cl, Cd = panel_method(X, Y, alpha_deg=alpha, kutta=True)
    
    # Split control points into upper and lower surfaces for plotting
    xc = 0.5 * (X[:-1] + X[1:])
    # Because of our Clockwise winding, the first half of the array is the lower surface, 
    # and the second half is the upper surface.
    mid_idx = len(xc) // 2
    
    plt.figure(figsize=(8, 5))
    # Aerodynamic standard: plot negative Cp (suction) pointing upwards
    plt.plot(xc[mid_idx:], Cp[mid_idx:], '-b', label='Upper Surface')
    plt.plot(xc[:mid_idx], Cp[:mid_idx], '-r', label='Lower Surface')
    plt.gca().invert_yaxis() # Flips the Y-axis upside down
    plt.title(f"Pressure Distribution (Cp) for NACA {code} at alpha = {alpha} degrees")
    plt.xlabel("Chordwise Position (x/c)")
    plt.ylabel("Pressure Coefficient (Cp)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(f"results/naca{code}_cp_dist.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Cp distribution plot (Calculated Lift Coeff: {Cl:.3f}).")
    
    # 3. Lift Sweep (Cl vs Angle of Attack)
    alphas = np.linspace(-5, 10, 10) # Test 10 angles between -5 and +10 degrees
    Cls = []
    
    print("Running lift sweep... this might take a few seconds.")
    for a in alphas:
        _, cl, _ = panel_method(X, Y, alpha_deg=a, kutta=True)
        Cls.append(cl)
        
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, Cls, '-o', color='blue', label='Panel Method')
    
    # Thin airfoil theory comparison: Cl = 2*pi*(alpha - alpha_L0)
    # alpha_L0 (zero-lift angle) is approximated from our panel method's intercept
    slope, intercept = np.polyfit(np.radians(alphas), Cls, 1)
    alpha_L0 = -intercept / slope
    cl_thin = 2 * np.pi * (np.radians(alphas) - alpha_L0)
    plt.plot(alphas, cl_thin, '--k', label='Thin Airfoil Theory (~6.28 slope)')
    
    plt.title(f"Lift Curve for NACA {code}")
    plt.xlabel("Angle of Attack (Degrees)")
    plt.ylabel("Lift Coefficient (Cl)")
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig(f"results/naca{code}_cl_sweep.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved Cl vs Alpha sweep plot.")
    print("Check your 'results/' folder to see the graphs!")

if __name__ == "__main__":
    run_demo()