import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad

# ---------------------------------- CONDUCTANCIA ----------------------------------

# ------------------ CÁLCULO NUMÉRICO ------------------

b = sp.symbols('b')  # ----> beta = 1/T con kb = 1
w = sp.symbols('w')  # ----> Frecuencia
Y = sp.symbols('Y')  # ----> Gamma
x = sp.symbols('x')  # ----> Energía sobre Gamma
n = sp.symbols('n')  # ----> Índice de la sumatoria
M = 20               # ----> Índice donde se trunca la sumatoria

# Distribución de Fermi y su derivada
P = 1 / (b*w - sp.I*sp.pi*(2*n - 1)) + 1 / (b*w + sp.I*sp.pi*(2*n - 1))
f = sp.Rational(1, 2) - sp.summation(P, (n, 1, M))
df = sp.diff(f, w)

# Densidad de estados
p = 1 / (sp.pi * ((w/Y - x)**2 + 1))

# Integrando de G/Go
integrando = -2 * sp.pi * Y * p * df
integrando_func = sp.lambdify([w, x, b, Y], integrando, "numpy")

# Cálculo numérico de G
def calcular_G(x_val, b_val, Y_val):
    resultado, _ = quad(lambda w: np.real(integrando_func(w, x_val, b_val, Y_val)), -np.inf, np.inf)
    return resultado

# ------------------ CÁLCULO ANALÍTICO (RESIDUOS) ------------------

w = sp.symbols('w', real=True)     # ----> Frecuencia
Y = sp.symbols('Y', real=True)     # ----> Gamma
x = sp.symbols('x', real=True)     # ----> Energía sobre Gamma
n = sp.symbols('n', integer=True)  # ----> Índice de la sumatoria
e = sp.symbols('e', real=True)     # ----> Energía
T = sp.symbols('T', real=True)     # ----> Temperatura
M = 20                             # ----> Índice donde se trunca la sumatoria

# Función de Green, derivada respecto a w y parte imaginaria
G = 1 / (sp.I*w - e + sp.I* Y)
dG_dw = sp.diff(G, w)
ImdG = sp.im(dG_dw)

# Reemplazar w con pi*T*(2n - 1)
ImdG = ImdG.subs(w, sp.pi*T*(2*n - 1))

# Residuo de la función para cada n
P = sp.I / (sp.I * w - T * sp.pi*sp.I*(2*n - 1))
polo = sp.pi*T*(2*n - 1)
residuo = sp.residue(P, w, polo)

# Cálculo de conductancia con residuos
C = 2*Y*T* sp.summation(residuo*ImdG, (n, 1, M)) * 2*sp.pi
C_func = sp.lambdify([e, Y, T], C, 'numpy')

# ------------------ GRAFICACIÓN ------------------

T_Y_vals = [0.5, 1, 2]              # ----> Valores de T/Y
x_vals = np.linspace(-10, 10, 600)  # ----> Valores de x (energía sobre gamma)
Y_val = 1                           # ----> Valor de Y

colors = plt.cm.Blues(np.linspace(0.5, 1, len(T_Y_vals)))

plt.figure(figsize=(8, 6))

# Método numérico
for idx, T_Y in enumerate(T_Y_vals):
    b_val = 1 / T_Y  # Relación inversa con T/Y
    G_vals = [calcular_G(x, b_val, Y_val) for x in x_vals]
    plt.plot(x_vals, G_vals, label=f'Numérico T$/\Delta$ = {T_Y}', color=colors[idx])

# Tomar menos valores para el residuo (Evitar saturación gráfica)
x_vals = np.linspace(-10, 10, 20)

# Método de residuos
for idx, T_Y in enumerate(T_Y_vals):
    C_vals = [C_func(x, Y_val, T_Y) for x in x_vals]
    plt.scatter(x_vals, C_vals, label=f'Residuo T$/\Delta$ = {T_Y}', color=colors[idx], marker='^')

plt.xlabel(r"$\epsilon/\Delta$", fontsize=14)
plt.ylabel("G/Go", fontsize=14)
plt.legend()
plt.grid()
plt.show()


# ---------------------------------- OCUPACIÓN DE ESTADOS ----------------------------------

# ------------------ CÁLCULO NUMÉRICO ------------------

b = sp.symbols('b')      # ----> Beta = 1/kbT con kb = 1
w = sp.symbols('w')      # ----> Frecuencia
Y = sp.symbols('Y')      # ----> Gamma
x = sp.symbols('x')      # ----> Energía sobre Gamma
n = sp.symbols('n')      # ----> Índice de sumatoria
M = 20                   # ----> Índice donde se trunca la sumatoria
e_Y = sp.symbols("e_Y")  # ----> Energía sobre Gamma

# Distribución de Fermi (f) y su derivada (df)
P = 1 / (b*w - sp.I*sp.pi*(2*n - 1)) + 1 / (b*w + sp.I*sp.pi*(2*n - 1))
f = sp.Rational(1, 2) - sp.summation(P, (n, 1, M))

# Densidad de estados
p = 1 / (sp.pi * ((w/Y - x)**2 + 1))

# Integrando de la ocupación
integrando_N = p * f
integrando_N_func = sp.lambdify([w, x, b, Y], integrando_N, "numpy")

# Cálculo numérico de N
def calcular_N(x_val, b_val, Y_val):
    resultado, _ = quad(lambda w: np.real(integrando_N_func(w, x_val, b_val, Y_val)), -np.inf, np.inf)
    return resultado/(2*np.pi)

# ------------------ CÁLCULO ANALÍTICO (RESIDUOS) ------------------

e = sp.symbols('e', real=True)     # ----> Energía
Y = sp.symbols('Y', real=True)     # ----> Gamma
T = sp.symbols('T', real=True)     # ----> Temperatura
w = sp.symbols('w', real=True)     # ----> Frecuencia
n = sp.symbols('n', integer=True)  # ----> Índice de la sumatoria
M = 20                             # ----> Índice donde se trunca la sumatoria

# Función espectral
p = Y / (sp.pi * ((sp.I * w - e)**2 + Y**2))

# Expresiones con residuos
Exp1 = p / (sp.I * w - sp.I * T * sp.pi * (2 * n - 1))
Exp2 = p / (sp.I * w + sp.I * T * sp.pi * (2 * n - 1))

# Cálculo de los residuos
Res1 = sp.residue(Exp1, w, sp.pi * T * (2 * n - 1)) + sp.residue(Exp1, w, Y - sp.I * e)
Res2 = sp.residue(Exp2, w, Y - sp.I * e)
ResT = Res1 + Res2
ResT = ResT.simplify()

# Ocupación de estados
N = 1/(4 * sp.pi) + T * sp.summation(ResT, (n, 1, M))
N_func = sp.lambdify([e, Y, T], N, 'numpy')

# ------------------ GRAFICACIÓN ------------------

T_Y_vals = [0.5, 1, 2]              # ----> Valores de T/Y
x_vals = np.linspace(-10, 10, 600)  # ----> Valores de x (energía sobre gamma)
Y_val = 1                           # ----> Valor de Y

colors = plt.cm.Blues(np.linspace(0.5, 1, len(T_Y_vals)))

plt.figure(figsize=(8, 6))

# Método numérico
for idx, T_Y in enumerate(T_Y_vals):
    N_vals = [calcular_N(x, 1/T_Y, Y_val) for x in x_vals]
    plt.plot(x_vals, N_vals, label=f'Numérico T$/\Delta$ = {T_Y}', color=colors[idx])

# Tomar menos valores para el residuo (Evitar saturación gráfica)
x_vals = np.linspace(-10, 10, 20)

# Método de residuos
for idx, T_Y in enumerate(T_Y_vals):
    N_vals = [N_func(x, Y_val, T_Y) for x in x_vals]
    plt.scatter(x_vals, N_vals, label=f'Residuo T$/\Delta$ = {T_Y}', color=colors[idx], marker='^')

plt.xlabel(r"$\epsilon/\Delta$", fontsize=14)
plt.ylabel("Ocupación de estados $n_{d\sigma}$", fontsize=14)
plt.legend()
plt.grid()
plt.show()
