#%% Import Libraries
import numpy as np  
from scipy.optimize import fsolve  
import sympy as sp  


#%%
# -----------------------------------------------------
# PROBLEM 1: Numerical Root-Finding for GPS Positioning
# -----------------------------------------------------

# Given constants and satellite data
c = 299792.458  # Speed of light in km/s
A = [15600, 18760, 17610, 19170]  # Satellite x-coordinates in km
B = [7540, 2750, 14630, 610]      # Satellite y-coordinates in km
C = [20140, 18610, 13480, 18390]  # Satellite z-coordinates in km
t = [0.07074, 0.07220, 0.07690, 0.07242]  # Signal travel times in seconds

# Function to define the residuals for the nonlinear system
def residuals(vars):
    """
    Residual function for GPS equations:
    sqrt((x - A_i)^2 + (y - B_i)^2 + (z - C_i)^2) - c * (t_i - d)
    """
    x, y, z, d = vars  # Unpack the unknowns
    res = []
    for i in range(4):  # Loop through the 4 satellites
        dist = np.sqrt((x - A[i])**2 + (y - B[i])**2 + (z - C[i])**2)
        res.append(dist - c * (t[i] - d))  # Append each residual
    return res

# Initial guess for (x, y, z, d)
initial_guess = [0, 0, 6370.0, 0]  # Receiver near Earth's surface and d = 0

# Solve
sol = fsolve(residuals, initial_guess)

# Print 
print("----- PROBLEM 1: Numerical Solution -----")
print(f"x = {sol[0]:.6f} km")
print(f"y = {sol[1]:.6f} km")
print(f"z = {sol[2]:.6f} km")
print(f"d = {sol[3]:.6e} seconds")
print("-----------------------------------------\n")

#%%
# ------------------------------------------------------------
# PROBLEM 2: Determinant-Based Analytical Approach for GPS
# ------------------------------------------------------------

# Define variables
x, y, z, d = sp.symbols('x y z d', real=True)

# Formulate the nonlinear equations
eqs = []
for i in range(4):
    eq = (x - A[i])**2 + (y - B[i])**2 + (z - C[i])**2 - (c * (t[i] - d))**2
    eqs.append(eq)

# Linearize the system
# Eliminate x^2, y^2, z^2 terms
lin_eqs = [sp.simplify(eqs[0] - eqs[i]) for i in range(1, 4)]

# Extract the coefficients of the linear equations
A_matrix, b_vector = sp.linear_eq_to_matrix(lin_eqs, [x, y, z, d])

# Split the coefficient matrix into components:
A_xyz = A_matrix[:, :3]  # Coefficients of x, y, z
A_d = A_matrix[:, 3]     # Coefficient of d

# Solve for x, y, z in terms of d
xyz_solution = A_xyz.LUsolve(b_vector - A_d * d)

# Simplify solutions for x, y, z as functions of d
x_d = sp.simplify(xyz_solution[0])
y_d = sp.simplify(xyz_solution[1])
z_d = sp.simplify(xyz_solution[2])

# Substitute x(d), y(d), z(d) into the first original equation
quadratic_eq_d = sp.simplify(eqs[0].subs({x: x_d, y: y_d, z: z_d}))

# Solve the resulting quadratic equation for d
coeffs_d = sp.Poly(quadratic_eq_d, d).all_coeffs()
d_solutions = sp.solve(quadratic_eq_d, d)

# Select the physically meaningful solution for d (real and close to zero)
d_final = None
for candidate in d_solutions:
    if candidate.is_real:
        d_final = candidate.evalf()
        break

# Compute final (x, y, z) by substituting d into x_d, y_d, z_d
x_final = x_d.subs(d, d_final).evalf()
y_final = y_d.subs(d, d_final).evalf()
z_final = z_d.subs(d, d_final).evalf()

# Print
print("----- PROBLEM 2: Analytical Solution -----")
print(f"x = {x_final:.6f} km")
print(f"y = {y_final:.6f} km")
print(f"z = {z_final:.6f} km")
print(f"d = {d_final:.6e} seconds")
print("-----------------------------------------")

#%%
# ------------------------------------------------------------
# PROBLEM 4 & 5: Conditioning Analysis of the GPS Problem
# ------------------------------------------------------------

# Constants
c = 299792.458  # Speed of light in km/s
rho = 26570  # Fixed satellite altitude in km
receiver_pos = np.array([0, 0, 6370])  # Receiver fixed at Earth's surface
d_initial = 0.0001  # Initial clock bias
perturbation = 1e-8  # Perturbation in seconds

# Function to compute satellite positions 
def compute_satellite_positions(phi, theta):
    A = [rho * np.cos(p) * np.cos(t) for p, t in zip(phi, theta)]
    B = [rho * np.cos(p) * np.sin(t) for p, t in zip(phi, theta)]
    C = [rho * np.sin(p) for p in phi]
    return np.array(A), np.array(B), np.array(C)

# Compute nominal ranges and travel times
def compute_nominal_values(A, B, C):
    R = np.sqrt((A - receiver_pos[0])**2 + (B - receiver_pos[1])**2 + (C - receiver_pos[2])**2)
    t_nominal = d_initial + R / c
    return R, t_nominal

# GPS residual function
def gps_residuals(vars, t, A, B, C):
    x, y, z, d = vars
    residuals = np.sqrt((x - A)**2 + (y - B)**2 + (z - C)**2) - c * (t - d)
    return residuals

# EMF computation function
def compute_emf(t_perturbed, t_nominal, A, B, C):
    initial_guess = [0, 0, 6370, d_initial]
    sol_nominal = fsolve(gps_residuals, initial_guess, args=(t_nominal, A, B, C))
    sol_perturbed = fsolve(gps_residuals, initial_guess, args=(t_perturbed, A, B, C))

    position_error = np.linalg.norm(np.array(sol_perturbed[:3]) - np.array(sol_nominal[:3]))
    input_error = np.linalg.norm(np.array(t_perturbed) - np.array(t_nominal)) * c
    return position_error / input_error

# Function to analyze EMF 
def analyze_emf(phi, theta):
    A, B, C = compute_satellite_positions(phi, theta)
    _, t_nominal = compute_nominal_values(A, B, C)

    emf_values = []
    for i in range(4):
        t_perturbed = t_nominal.copy()
        t_perturbed[i] += perturbation
        emf = compute_emf(t_perturbed, t_nominal, A, B, C)
        emf_values.append(emf)

    return emf_values

# Loose satellite configuration
phi_loose = [np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 8]
theta_loose = [0, np.pi / 2, np.pi, 3 * np.pi / 2]
emf_loose = analyze_emf(phi_loose, theta_loose)

# Tightly grouped satellite configuration
def tightly_grouped_coordinates(phi_base, theta_base, perturb=0.05):
    np.random.seed(42)  # For reproducibility
    phi_tight = [phi_base * (1 + np.random.uniform(-perturb, perturb)) for _ in range(4)]
    theta_tight = [theta_base * (1 + np.random.uniform(-perturb, perturb)) for _ in range(4)]
    return phi_tight, theta_tight

phi_base, theta_base = np.pi / 4, np.pi / 2
phi_tight, theta_tight = tightly_grouped_coordinates(phi_base, theta_base)
emf_tight = analyze_emf(phi_tight, theta_tight)

# Print results
print("----- PROBLEM 4 & 5: CONDITIONING ANALYSIS COMPARISON -----")
print("Loose Satellites:")
for i, emf in enumerate(emf_loose):
    print(f"EMF for perturbation in t_{i+1}: {emf:.6f}")
print(f"Maximum EMF (Loose): {max(emf_loose):.6f}\n")

print("Tightly Grouped Satellites:")
for i, emf in enumerate(emf_tight):
    print(f"EMF for perturbation in t_{i+1}: {emf:.6f}")
print(f"Maximum EMF (Tight): {max(emf_tight):.6f}")
print("--------------------------------------------")
