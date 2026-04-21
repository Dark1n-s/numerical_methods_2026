import random

def mat_vec_mult(A, X):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * X[j] for j in range(len(X)))
    return res

def vector_norm(V):
    return max(abs(x) for x in V)

def write_matrix(mat, filename):
    with open(filename, 'w') as f:
        for row in mat:
            f.write("\t".join(f"{val:.6f}" for val in row) + "\n")

def write_vector(vec, filename):
    with open(filename, 'w') as f:
        for val in vec:
            f.write(f"{val:.15f}\n")

def read_matrix(filename):
    mat = []
    with open(filename, 'r') as f:
        for line in f:
            mat.append([float(x) for x in line.split()])
    return mat

def read_vector(filename):

    vec = []
    with open(filename, 'r') as f:
        for line in f:
            vec.append(float(line.strip()))
    return vec

# ---------------------------------------------------------
# Алгоритми LU
# ---------------------------------------------------------

def lu_decomposition(A):
    n = len(A)
    L = [[0.0] * n for _ in range(n)]
    U = [[0.0] * n for _ in range(n)]
    for i in range(n):
        U[i][i] = 1.0

    for k in range(n):

        for i in range(k, n):
            s = sum(L[i][j] * U[j][k] for j in range(k))
            L[i][k] = A[i][k] - s
            
        for j in range(k + 1, n):
            s = sum(L[k][p] * U[p][j] for p in range(k))
            U[k][j] = (A[k][j] - s) / L[k][k]
            
    return L, U

def solve_lu(L, U, B):
    n = len(L)
    
    Z = [0.0] * n
    for i in range(n):
        s = sum(L[i][j] * Z[j] for j in range(i))
        Z[i] = (B[i] - s) / L[i][i]

    X = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = sum(U[i][j] * X[j] for j in range(i + 1, n))
        X[i] = Z[i] - s
        
    return X


def main():
    n = 100
    
    A_generated = [[random.uniform(1.0, 10.0) for _ in range(n)] for _ in range(n)]
    write_matrix(A_generated, "matrix_A.txt")
    
    X_exact = [2.5] * n
    write_vector(X_exact, "vector_X_exact.txt")
    
    B_generated = mat_vec_mult(A_generated, X_exact)
    write_vector(B_generated, "vector_B.txt")
    
    A = read_matrix("matrix_A.txt")
    B = read_vector("vector_B.txt")
    
    L, U = lu_decomposition(A)
    write_matrix(L, "matrix_L.txt")
    write_matrix(U, "matrix_U.txt")
    
    X_0 = solve_lu(L, U, B)

    B_calc = mat_vec_mult(A, X_0)
    diff = [B_calc[i] - B[i] for i in range(n)]
    eps_initial = vector_norm(diff)
    print(f"Початкова похибка (eps): {eps_initial:.5e}")
    
    eps_target = 1e-14
    X_final = X_0.copy() 
    iterations = 0

    while True:
        iterations += 1
        B_k = mat_vec_mult(A, X_final)
        R = [B[i] - B_k[i] for i in range(n)]
        delta_X = solve_lu(L, U, R)
        
        X_final = [X_final[i] + delta_X[i] for i in range(n)]
        
        norm_delta_X = vector_norm(delta_X)
        if norm_delta_X <= eps_target:
            break
            
        if iterations > 70:
            print("Досягнуто ліміту ітерацій.")
            break

    write_vector(X_final, "vector_X_calculated.txt")

    print("-" * 40)
    print("РЕЗУЛЬТАТИ:")
    print(f"Необхідна кількість ітерацій: {iterations}")
    print(f"Кінцева норма похибки (||delta X||): {norm_delta_X:.5e}")
 
if __name__ == "__main__":
    main()