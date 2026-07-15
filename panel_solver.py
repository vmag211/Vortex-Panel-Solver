import numpy as np

def naca4_coordinates(code, n_panels=160, chord=1.0):
    """
    Generates closed-loop coordinates for a NACA 4-digit airfoil.
    Uses cosine spacing to cluster points near the leading and trailing edges.
    Winding direction: Clockwise (CW) from lower trailing edge -> LE -> upper trailing edge.
    """
    # Parse the 4-digit code (e.g., '2412')
    m = int(code[0]) / 100.0   # Maximum camber (percentage of chord)
    p = int(code[1]) / 10.0    # Position of max camber (tenths of chord)
    t = int(code[2:]) / 100.0  # Maximum thickness (percentage of chord)

    # Cosine spacing for the x-coordinates along the chord (0 to 1)
    # beta goes from 0 to pi, clustering points at both ends
    beta = np.linspace(0, np.pi, n_panels // 2 + 1)
    x_c = chord * 0.5 * (1 - np.cos(beta))

    # Thickness distribution equation
    # We use -0.1036 instead of -0.1015 so the trailing edge closes perfectly to 0
    y_t = 5 * t * chord * (0.2969 * np.sqrt(x_c/chord) - 0.1260 * (x_c/chord) - 
                           0.3516 * (x_c/chord)**2 + 0.2843 * (x_c/chord)**3 - 
                           0.1036 * (x_c/chord)**4)

    # Camber line and its derivative (slope)
    y_c = np.zeros_like(x_c)
    dyc_dx = np.zeros_like(x_c)

    for i, x in enumerate(x_c):
        if p == 0: # Symmetric airfoil (no camber)
            continue
        if x < p * chord:
            y_c[i] = m * (x / p**2) * (2*p - x/chord)
            dyc_dx[i] = (2*m / p**2) * (p - x/chord)
        else:
            y_c[i] = m * ((chord - x) / (1-p)**2) * (1 + x/chord - 2*p)
            dyc_dx[i] = (2*m / (1-p)**2) * (p - x/chord)

    # Angle of the camber line
    theta = np.arctan(dyc_dx)

    # Calculate Upper (U) and Lower (L) surface points
    x_U = x_c - y_t * np.sin(theta)
    y_U = y_c + y_t * np.cos(theta)
    x_L = x_c + y_t * np.sin(theta)
    y_L = y_c - y_t * np.cos(theta)

    # Assemble into a single Clockwise (CW) loop
    # Lower surface goes from LE (index 0) to TE (index -1). We reverse it to start at TE.
    # Upper surface goes from LE to TE. We drop the duplicate LE point at index 0.
    X = np.concatenate([x_L[::-1], x_U[1:]])
    Y = np.concatenate([y_L[::-1], y_U[1:]])

    return X, Y

def panel_method(X, Y, alpha_deg, Vinf=1.0, kutta=True):
    """
    Solves the constant-strength source + shared vortex panel method.
    Returns Cp (pressure coeff), Cl (lift coeff), Cd (drag coeff).
    """
    alpha = np.radians(alpha_deg)
    N = len(X) - 1 # Number of panels is one less than number of points

    # Pre-allocate panel geometry arrays
    xc = np.zeros(N)  # Control point x (center of panel)
    yc = np.zeros(N)  # Control point y (center of panel)
    S = np.zeros(N)   # Panel length
    phi = np.zeros(N) # Panel orientation angle

    # Calculate panel geometries
    for i in range(N):
        xc[i] = 0.5 * (X[i] + X[i+1])
        yc[i] = 0.5 * (Y[i] + Y[i+1])
        dx = X[i+1] - X[i]
        dy = Y[i+1] - Y[i]
        S[i] = np.hypot(dx, dy)
        phi[i] = np.arctan2(dy, dx)

    # Normal vector angle (pointing strictly outward due to CW winding)
    beta = phi + np.pi/2

    # Influence matrices mapping how every panel affects every control point
    A = np.zeros((N, N)) # Normal velocity from sources
    B_vortex = np.zeros(N) # Normal velocity from the shared vortex
    C = np.zeros((N, N)) # Tangential velocity from sources
    D_vortex = np.zeros(N) # Tangential velocity from the shared vortex

    for i in range(N):
        for j in range(N):
            if i == j:
                # A panel's source creates a normal velocity of 0.5 on itself
                A[i,j] = 0.5
                C[i,j] = 0.0
                # A panel's vortex creates a tangential velocity of 0.5 on itself
                B_vortex[i] += 0.0
                D_vortex[i] += 0.5
                continue

            # Translate control point i to panel j's local coordinate system
            dx = xc[i] - X[j]
            dy = yc[i] - Y[j]
            cos_p = np.cos(phi[j])
            sin_p = np.sin(phi[j])
            x_loc =  dx * cos_p + dy * sin_p
            y_loc = -dx * sin_p + dy * cos_p

            # Calculate local radii and angles from panel edges to control point
            r1 = np.hypot(x_loc, y_loc)
            r2 = np.hypot(x_loc - S[j], y_loc)
            th1 = np.arctan2(y_loc, x_loc)
            th2 = np.arctan2(y_loc, x_loc - S[j])

            # Ensure angle difference doesn't flip incorrectly across the -pi/pi boundary
            dtheta = th2 - th1
            dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi

            # Local induced velocities per unit strength
            u_s = (1.0 / (2*np.pi)) * np.log(r1 / r2)
            v_s = (1.0 / (2*np.pi)) * dtheta
            
            u_v = v_s
            v_v = -u_s

            # Rotate induced velocities back to global coordinates
            U_s = u_s * cos_p - v_s * sin_p
            V_s = u_s * sin_p + v_s * cos_p
            U_v = u_v * cos_p - v_v * sin_p
            V_v = u_v * sin_p + v_v * cos_p

            # Project velocities onto the normal and tangent of panel i
            nx, ny = np.cos(beta[i]), np.sin(beta[i])
            tx, ty = np.cos(phi[i]), np.sin(phi[i])

            A[i,j] = U_s * nx + V_s * ny
            C[i,j] = U_s * tx + V_s * ty
            B_vortex[i] += U_v * nx + V_v * ny
            D_vortex[i] += U_v * tx + V_v * ty

    # Freestream velocity components on each panel
    Vinf_n = Vinf * np.cos(beta - alpha)
    Vinf_t = Vinf * np.cos(phi - alpha)

    if not kutta:
        # Solve without vortex (pure source panels)
        lambdas = np.linalg.solve(A, -Vinf_n)
        gamma = 0.0
        Vt = Vinf_t + C @ lambdas
    else:
        # Build the (N+1) x (N+1) matrix to include the Kutta condition
        # The Kutta condition forces tangential velocity at the two TE panels to sum to zero
        M = np.zeros((N+1, N+1))
        RHS = np.zeros(N+1)

        M[:N, :N] = A
        M[:N, N] = B_vortex
        RHS[:N] = -Vinf_n

        # The last equation is the Kutta condition
        M[N, :N] = C[0, :] + C[-1, :]
        M[N, N] = D_vortex[0] + D_vortex[-1]
        RHS[N] = -(Vinf_t[0] + Vinf_t[-1])

        # Solve for N source strengths (lambdas) + 1 vortex strength (gamma)
        sol = np.linalg.solve(M, RHS)
        lambdas = sol[:N]
        gamma = sol[N]

        # Calculate final tangential velocity at each panel
        Vt = Vinf_t + C @ lambdas + D_vortex * gamma

    # Pressure coefficient (Bernoulli's principle)
    Cp = 1.0 - (Vt / Vinf)**2

    # Calculate forces integrating pressure over the geometry
    Fx = np.sum(-Cp * S * np.cos(beta))
    Fy = np.sum(-Cp * S * np.sin(beta))

    # Rotate forces into the freestream frame to get Lift and Drag
    Cl = Fy * np.cos(alpha) - Fx * np.sin(alpha)
    Cd = Fx * np.cos(alpha) + Fy * np.sin(alpha)

    return Cp, Cl, Cd