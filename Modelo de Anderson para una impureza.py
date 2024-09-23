import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.integrate import quad

# Definición de símbolos para las ecuaciones
b = sp.symbols('b')      # ----> Beta = 1/kbT con kb = 1
w = sp.symbols('w')      # ----> Frecuencia
Y = sp.symbols('Y')      # ----> Gamma
x = sp.symbols('x')      # ----> Energía sobre Gamma
n = sp.symbols('n')      # ----> Índice de sumatoria
M = 20                   # ----> Índice donde se trunca la sumatoria
e_Y = sp.symbols("e_Y")  # ----> Energía sobre Gamma

# ------------------ FUNCIONES PARA CONDUCTANCIA LINEAL Y OCUPACIÓN ------------------

# Distribución de Fermi (f) y su derivada (df)
P = 1 / (b*w - sp.I*sp.pi*(2*n - 1)) + 1 / (b*w + sp.I*sp.pi*(2*n - 1))
f = sp.Rational(1, 2) - sp.summation(P, (n, 1, M))
df = sp.diff(f, w)

# Densidad de estados para la conductancia y ocupación
p = 1 / (sp.pi * ((w/Y - x)**2 + 1))

# Integrandos para conductancia (C) y ocupación (N)
integrando_G = -2 * sp.pi * Y * p * df
integrando_N = p * f

integrando_G_func = sp.lambdify([w, x, b, Y], integrando_G, "numpy")
integrando_N_func = sp.lambdify([w, x, b, Y], integrando_N, "numpy")

# Función para calcular la conductancia G/Go
def calcular_G(x_val, b_val, Y_val):
    resultado, _ = quad(lambda w: np.real(integrando_G_func(w, x_val, b_val, Y_val)), -np.inf, np.inf)
    return resultado

# Función para calcular la ocupación N
def calcular_N(x_val, b_val, Y_val):
    resultado, _ = quad(lambda w: np.real(integrando_N_func(w, x_val, b_val, Y_val)), -np.inf, np.inf)
    return resultado

# Valores de T/Y y x
T_Y_values = [0.5, 1, 2]
x_vals = np.linspace(-10, 10, 600)

colors = cm.Blues(np.linspace(0.5, 1, len(T_Y_values)))

# Gráfica de conductancia G/Go
plt.figure(figsize=(8, 6))
for idx, T_Y in enumerate(T_Y_values):
    b_val = 1 / T_Y  # Relación inversa con T/Y
    G_vals = [calcular_G(x, b_val, 1) for x in x_vals]  # Y = 1
    plt.plot(x_vals, G_vals, label=f'T$/\Delta$ = {T_Y}', color=colors[idx])
plt.xlabel(r"$\epsilon/\Delta$", fontsize=14)
plt.ylabel("G/Go", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# Gráfica de ocupación
plt.figure(figsize=(8, 6))
for idx, T_Y in enumerate(T_Y_values):
    b_val = 1 / T_Y
    N_vals = [calcular_N(x, b_val, 1) for x in x_vals]
    plt.plot(x_vals, N_vals, label=f'T$/\Delta$ = {T_Y}', color=colors[idx])
plt.xlabel(r"$\epsilon/\Delta$", fontsize=14)
plt.ylabel("$n_{d\sigma}$", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

# ------------------ DENSIDAD DE ESTADOS ------------------

# Densidad de estados
p = 1 / (sp.pi * ((x - e_Y)**2 + 1))

pfunc = sp.lambdify([x, e_Y], p, 'numpy')

# Valores de e/Y y x
vece_Y = [0, 1, 2]
vecx = np.linspace(-4, 6, 1000)

# Gráfica de densidad de estados
plt.figure(figsize=(8, 6))
for idx, i in enumerate(vece_Y):
    vecp = [pfunc(xd, i) for xd in vecx]
    plt.plot(vecx, vecp, label=f'$\epsilon / \Delta$ = {i}', color=colors[idx])
plt.xlabel(r'$\omega / \Delta$', fontsize=14)
plt.ylabel('Densidad de estados', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True)
plt.show()

