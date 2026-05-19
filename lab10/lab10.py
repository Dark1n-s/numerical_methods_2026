import numpy as np
import matplotlib.pyplot as plt

def f(x, y):
    return -y + x + 1
def y_exact(x):
    return x + np.exp(-x)
def rk4_step(f, x, y, h):
    k1 = f(x, y)
    k2 = f(x + h/2, y + h*k1/2)
    k3 = f(x + h/2, y + h*k2/2)
    k4 = f(x + h, y + h*k3)
    return y + (h/6) * (k1+ 2*k2 + 2*k3 + k4)
def adams_pc_2(f, x0, y0, a, b, h):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    y_pr = np.zeros(len(x))
    
    y[0] = y0
    if len(x) > 1:
        y[1] = rk4_step(f, x[0], y[0], h)
        
    for n in range(1, len(x) - 1):
        y_pr[n+1] = y[n] + (h/2) * (3*f(x[n], y[n]) - f(x[n-1], y[n-1]))
        y[n+1] = y[n] + (h/2) * (f(x[n+1], y_pr[n+1]) + f(x[n], y[n]))
        
    return x, y, y_pr
def adams_pc_2_auto(f, x0, y0, a, b, tol):
    x = [a]
    y = [y0]
    h_vals = []
    h = 0.1 
    
    y1 = rk4_step(f, x[0], y[0], h)
    x.append(a + h)
    y.append(y1)
    h_vals.append(h)
    
    n = 1
    while x[-1] < b:
        h = min(h, b - x[-1])
        y_pr = y[n] + (h/2) * (3*f(x[n], y[n]) - f(x[n-1], y[n-1]))
        y_kor = y[n] + (h/2) * (f(x[-1] + h, y_pr) + f(x[n], y[n]))
        
        err = abs(y_kor - y_pr) / 6
        
        if err <= tol:
            x.append(x[-1] + h)
            y.append(y_kor)
            h_vals.append(h)
            n += 1
            if err < tol / 10:
                h *= 2
        else:
            h /= 2
            
    return np.array(x), np.array(y), np.array(h_vals)

def rk4_method(f, x0, y0, a, b, h):
    x = np.arange(a, b + h, h)
    y = np.zeros(len(x))
    y[0] = y0
    for n in range(len(x) - 1):
        y[n+1] = rk4_step(f, x[n], y[n], h)
    return x, y

def runge_error_rk4(f, x_val, y_val, h):
    y_h = rk4_step(f, x_val, y_val, h)
    y_h2_half = rk4_step(f, x_val, y_val, h/2)
    y_h2 = rk4_step(f, x_val + h/2, y_h2_half, h/2)
    err = (16/15) * abs(y_h2 - y_h)
    return y_h2, err

def rk4_auto(f, x0, y0, a, b, tol):
    x = [a]
    y = [y0]
    h_vals = []
    h = 0.1
    
    while x[-1] < b:
        h = min(h, b - x[-1])
        y_next, err = runge_error_rk4(f, x[-1], y[-1], h)
        
        if err <= tol:
            x.append(x[-1] + h)
            y.append(y_next)
            h_vals.append(h)
            if err < tol / 64:
                h *= 2
        else:
            h /= 2
            
    return np.array(x), np.array(y), np.array(h_vals)


a, b = 0, 5
x0, y0 = a, 1
h_const = 0.1
tol = 1e-4

x_adams, y_adams, y_pr_adams = adams_pc_2(f, x0, y0, a, b, h_const)
err_exact_adams = abs(y_adams - y_exact(x_adams))
err_est_adams = abs(y_adams - y_pr_adams) / 6

x_auto_adams, y_auto_adams, h_auto_adams = adams_pc_2_auto(f, x0, y0, a, b, tol)

print("----------------------------------------------------------")
print("ЧАСТИНА 1: МЕТОД ПРОГНОЗУ ТА КОРЕКЦІЇ АДАМСА (h = 0.1)")
print("----------------------------------------------------------")
print(f"{'x':<8}{'y_Адамса':<15}{'y_Точне':<15}{'Точна похибка':<15}{'Оцінка похибки':<15}")
for i in range(0, len(x_adams), 5):
    print(f"{x_adams[i]:<8.2f}{y_adams[i]:<15.6f}{y_exact(x_adams[i]):<15.6f}{err_exact_adams[i]:<15.2e}{err_est_adams[i]:<15.2e}")

print(f"\n[Автоматичний крок Адамса]: Розраховано точок: {len(x_auto_adams)}, мін. крок: {min(h_auto_adams):.4f}, макс. крок: {max(h_auto_adams):.4f}\n")


h_rk = 1e-2
x_rk, y_rk = rk4_method(f, x0, y0, a, b, h_rk)
err_exact_rk = abs(y_rk - y_exact(x_rk))

err_runge_rk = []
for i in range(len(x_rk)-1):
    _, err = runge_error_rk4(f, x_rk[i], y_rk[i], h_rk)
    err_runge_rk.append(err)
err_runge_rk.insert(0, 0)

x_auto_rk, y_auto_rk, h_auto_rk = rk4_auto(f, x0, y0, a, b, tol)

print("----------------------------------------------------------")
print("ЧАСТИНА 2: МЕТОД РУНГЕ-КУТТА 4-ГО ПОРЯДКУ (h = 0.01)")
print("----------------------------------------------------------")
print(f"{'x':<8}{'y_Рунге-Кутта':<15}{'y_Точне':<15}{'Точна похибка':<15}{'Оцінка Рунге':<15}")
for i in range(0, len(x_rk), 50):
    print(f"{x_rk[i]:<8.2f}{y_rk[i]:<15.6f}{y_exact(x_rk[i]):<15.6f}{err_exact_rk[i]:<15.2e}{err_runge_rk[i]:<15.2e}")

print(f"\n[Автоматичний крок РК4]: Розраховано точок: {len(x_auto_rk)}, мін. крок: {min(h_auto_rk):.4f}, макс. крок: {max(h_auto_rk):.4f}\n")
print("==========================================================")


plt.figure(figsize=(15, 10))

plt.subplot(3, 2, 1)
plt.plot(x_adams, err_exact_adams, label='Точна похибка')
plt.title('Ч.1: Локальна похибка Адамса ')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(x_adams, err_est_adams, label='Оцінка похибки (y_kor - y_pr)')
plt.title('Ч.1: Оцінка похибки Адамса')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(x_auto_adams[:-1], h_auto_adams, label='Крок h(x)')
plt.title('Ч.1: Автоматичний вибір кроку Адамса')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(x_rk, err_exact_rk, color='red', label='Точна похибка')
plt.title('Ч.2: Локальна похибка Рунге-Кутта ')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(x_rk, err_runge_rk, color='red', label='Оцінка Рунге')
plt.title('Ч.2: Оцінка похибки Рунге-Кутта (Правило Рунге)')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(x_auto_rk[:-1], h_auto_rk, color='red', label='Крок h(x)')
plt.title('Ч.2: Автоматичний вибір кроку Рунге-Кутта')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()