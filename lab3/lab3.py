import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



df = pd.read_csv('data.csv')
x = df['Month'].values.astype(np.float64)
y = df['Temp'].values.astype(np.float64)


def form_matrix(x, m):
    A = np.zeros((m + 1, m + 1), dtype=np.float64)
    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))
    return A

def form_vector(x, y, m):
    b = np.zeros(m + 1, dtype=np.float64)
    for i in range(m + 1):
        b[i] = np.sum(y * (x ** i))
    return b

def gauss_solve(A_in, b_in):
    A = A_in.copy()
    b = b_in.copy()
    n = len(b)
    
    # Прямий хід
    for k in range(n - 1):
        max_row = np.argmax(np.abs(A[k:n, k])) + k
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]
        
        for i in range(k + 1, n):
            if A[k, k] == 0: continue
            factor = A[i, k] / A[k, k]
            A[i, k:] = A[i, k:] - factor * A[k, k:]
            b[i] = b[i] - factor * b[k]
            
    # Зворотній хід
    x_sol = np.zeros(n, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if A[i, i] != 0:
            x_sol[i] = (b[i] - np.sum(A[i, i+1:] * x_sol[i+1:])) / A[i, i]
    return x_sol

def polynomial(x, coef):
    y_poly = np.zeros_like(x, dtype=np.float64)
    for i in range(len(coef)):
        y_poly += coef[i] * (x ** i)
    return y_poly

def variance(y_true, y_approx):
    return np.sum((y_true - y_approx) ** 2) / (len(y_true) + 1)


max_degree = 10
variances = []
best_var = float('inf')
optimal_m = 1

print("--- Дисперсії для різних степенів ---")
for m in range(1, max_degree + 1):
    A = form_matrix(x, m)
    b_vec = form_vector(x, y, m)
    coef = gauss_solve(A, b_vec)
    y_approx = polynomial(x, coef)
    var = variance(y, y_approx)
    variances.append(var)
    
    print(f"Ступінь m={m}: дисперсія = {var:.4f}")
    
    if var < best_var:
        best_var = var
        optimal_m = m

print(f"\n=> Оптимальний ступінь полінома за мінімумом дисперсії: {optimal_m}")
A = form_matrix(x, optimal_m)
b_vec = form_vector(x, y, optimal_m)
coef = gauss_solve(A, b_vec)
y_approx = polynomial(x, coef)

x_future = np.array([25, 26, 27], dtype=np.float64)
y_future = polynomial(x_future, coef)
print(f"=> Прогноз температур на 25, 26, 27 місяці: {np.round(y_future, 2)}")


error = np.abs(y - y_approx)
err_df = pd.DataFrame({
    'Month': x, 
    'Actual Temp': y, 
    'Approximated Temp': np.round(y_approx, 2), 
    'Absolute Error': np.round(error, 2)
})
print("\n--- Таблиця похибок апроксимації (перші 10 значень) ---")
print(err_df.head(10))


plt.figure(figsize=(16, 10))


plt.subplot(2, 2, 1)
plt.plot(x, y, 'o', label='Фактичні дані', color='blue')
x_dense = np.linspace(min(x), max(x), 100)
plt.plot(x_dense, polynomial(x_dense, coef), '-', label=f'Апроксимація (m={optimal_m})', color='green')
plt.plot(x_future, y_future, 'X', color='red', markersize=8, label='Прогноз')
plt.title('Апроксимація температури')
plt.grid(True); plt.legend()

plt.subplot(2, 2, 2)
m_values = range(1, max_degree + 1)
plt.plot(m_values, variances, marker='s', color='orange', linestyle='--')
plt.title('Залежність дисперсії від степені m')
plt.xlabel('Ступінь многочлена m')
plt.ylabel('Дисперсія')
plt.xticks(m_values)
plt.grid(True)


plt.subplot(2, 2, 3)
for m in range(1, max_degree + 1):
    A_temp = form_matrix(x, m)
    b_temp = form_vector(x, y, m)
    coef_temp = gauss_solve(A_temp, b_temp)
    y_approx_temp = polynomial(x, coef_temp)
    error_temp = np.abs(y - y_approx_temp)
    plt.plot(x, error_temp, '-o', label=f'm={m}', alpha=0.7, markersize=4)

plt.title('Похибки для випадків m=1...10')
plt.xlabel('Вузли (Місяці)')
plt.ylabel('Похибка')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')

plt.tight_layout()

plt.show()
