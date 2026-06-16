import sympy as sp

# Define a function
x = sp.Symbol('x')
f = sp.exp(-x)

# Compute indefinite integral
indefinite_integral = sp.integrate(f, x)
print("Indefinite integral: ", indefinite_integral)

# Compute definite integral
definite_integral = sp.integrate(f, (x, 0, sp.oo))
print("Definite Integral: ", definite_integral)

f = x**2
definite_integral = sp.integrate(f, (x, 0, 2))
indefinite_integral = sp.integrate(f, x)
print("Definite Integral:", definite_integral)
print("Indefinite Integral:", indefinite_integral)
