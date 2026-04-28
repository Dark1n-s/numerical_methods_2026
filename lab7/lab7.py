import random


#--------------------------------------------------Basic-------------------------------------------------------------#


def write_matrix(mat, filename):
    with open(filename, 'w') as f:
        for row in mat:
            f.write("\t".join(f"{val:.3f}" for val in row) + "\n")

def write_vector(vec, filename):
    with open(filename, 'w') as f:
        for val in vec:
            f.write(f"{val:.6f}\n")

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

def mat_vec_mult(A, X):
    n = len(A)
    res = [0.0] * n
    for i in range(n):
        res[i] = sum(A[i][j] * X[j] for j in range(len(X)))
    return res

def vector_norm(V):
    return max(abs(x) for x in V)

def matrix_norm(A):
    return max(sum(abs(x) for x in row) for row in A)


#--------------------------------------------------Simple-------------------------------------------------------------#


def simple_iteration_method(A, B, X0, eps):
    n = len(A)
    X_k = X0.copy()
    iterations = 0

    norm_A = matrix_norm(A)
    tau = 1.0 / norm_A 
    
    while True:
        iterations += 1
        X_next = [0.0] * n

        for i in range(n):
            sum_Ax = sum(A[i][j] * X_k[j] for j in range(n))
            X_next[i] = X_k[i] - tau * sum_Ax + tau * B[i]
            
        diff = [X_next[i] - X_k[i] for i in range(n)]
        if vector_norm(diff) <= eps:
            break
            
        X_k = X_next
        if iterations > 10000:
            break
            
    return X_next, iterations


#--------------------------------------------------Jacobi-------------------------------------------------------------#


def jacobi_method(A, B, X0, eps):
    n = len(A)
    X_k = X0.copy()
    iterations = 0
    
    while True:
        iterations += 1
        X_next = [0.0] * n
        
        for i in range(n):
            s = sum(A[i][j] * X_k[j] for j in range(n) if j != i)
            X_next[i] = (B[i] - s) / A[i][i]
            
        diff = [X_next[i] - X_k[i] for i in range(n)]
        if vector_norm(diff) <= eps: 
            break
            
        X_k = X_next
        if iterations > 10000:
            break
            
    return X_next, iterations


#--------------------------------------------------Seidel-------------------------------------------------------------#


def seidel_method(A, B, X0, eps):
    n = len(A)
    X_k = X0.copy()
    iterations = 0
    
    while True:
        iterations += 1
        X_next = X_k.copy()
        
        for i in range(n):
            s1 = sum(A[i][j] * X_next[j] for j in range(i))
            s2 = sum(A[i][j] * X_k[j] for j in range(i + 1, n))
            
            X_next[i] = (B[i] - s1 - s2) / A[i][i]
            
        diff = [X_next[i] - X_k[i] for i in range(n)]
        if vector_norm(diff) <= eps: 
            break
            
        X_k = X_next
        if iterations > 10000:
            break
            
    return X_next, iterations


#--------------------------------------------------Interface-------------------------------------------------------------#


def main():
    n = 100 
    eps_target = 1e-14 
    
    A_generated = [[random.uniform(1.0, 10.0) for _ in range(n)] for _ in range(n)]
    
    for i in range(n):
        row_sum = sum(abs(A_generated[i][j]) for j in range(n) if i != j)
        A_generated[i][i] += row_sum + 200.0 
        
    write_matrix(A_generated, "matrix_A_lab8.txt")
    

    X_exact = [2.5] * n
    
    B_generated = mat_vec_mult(A_generated, X_exact)
    write_vector(B_generated, "vector_B_lab8.txt") 
    
    A = read_matrix("matrix_A_lab8.txt")
    B = read_vector("vector_B_lab8.txt")
    
   
    write_vector(X_exact, "vector_X_exact.txt")
    print("   Вектор точного розв'язку збережено у 'vector_X_exact.txt'")
    
    A = read_matrix("matrix_A_lab8.txt")
    B = read_vector("vector_B_lab8.txt")
    
    X_0 = [1.0] * n
    
    print(f"\nТочність (eps) = {eps_target}")
    print("-" * 50)
    
    X_simple, iter_simple = simple_iteration_method(A, B, X_0, eps_target)
    write_vector(X_simple, "vector_X_simple.txt")
    print(f"Метод простої ітерації: {iter_simple} ітерацій. (збережено у vector_X_simple.txt)")
    
    X_jacobi, iter_jacobi = jacobi_method(A, B, X_0, eps_target)
    write_vector(X_jacobi, "vector_X_jacobi.txt")
    print(f"Метод Якобі:           {iter_jacobi} ітерацій. (збережено у vector_X_jacobi.txt)")
    
    X_seidel, iter_seidel = seidel_method(A, B, X_0, eps_target)
    write_vector(X_seidel, "vector_X_seidel.txt")
    print(f"Метод Гауса-Зейделя:   {iter_seidel} ітерацій. (збережено у vector_X_seidel.txt)")
   
   
    print("-" * 50)

if __name__ == "__main__":
    main()