import time

import matplotlib.pyplot as plt
import numpy as np

try:
    np.set_printoptions(precision=3, linewidth=150)
    start = time.time()
    n = int(input())
    while n < 4:
        n = int(input('Введите количество строк(столбцов) квадратной матрицы > 3'))
    k = int(input())
    A = np.random.randint(-10, 10, (n, n))                                                                              # задаем матрицу A
    F = np.copy(A)                                                                                                      # задаем матрицу F
    print('A')
    print(A)
    s, r = 0, 1                                                                                                         # введем переменную для подсчета количество нулей в C в нечетных столбцах
    for i in range(n):
        for j in range(n):
            if i < (n // 2) and j > (n // 2 - (n - 1) % 2):
                if j % 2 == 0 and A[i][j] == 0:
                    s += 1
                if i == 0 or i == (n // 2 - 1) or j == (n // 2 + n % 2) or j == (n - 1):
                    r *= int(A[i][j])
    print(s, r)
    if s > r:
        for i in range(n // 2):                                                                                         # если нулей больше то мы симметрично меняем B и C
            F[i] = F[i][::-1]
    else:                                                                                                               # иначе меняем B и E местами несимметрично
        for i in range(n // 2):
            for j in range(n // 2):
                F[i][j], F[i + n // 2 + n % 2][j + n // 2 + n % 2] = F[i + n // 2 + n % 2][j + n // 2 + n % 2], F[i][j]
    print('F')
    print(F)
    if np.linalg.det(A) > sum(np.diagonal(F)):
        if np.linalg.det(A) == 0:
            print("Матрица А вырожденная, дальнейшие вычисления невозможны")
        else:
            print(f"(A-1)*AT – K * F\n{np.matmul(np.linalg.inv(A), np.transpose(A)) - k * F}")
    else:
        if np.linalg.det(F) == 0 or np.linalg.det(np.tril(A)) == 0:
            print("Матрица F или G вырожденная, дальнейшие вычисления невозможны")
        else:
            print (f'G\n{np.tril(A)}')
            print(f"(A +(G-1)-(F-1))*K\n{(A + np.linalg.inv(np.tril(A)) - np.linalg.inv(F)) * k}")
except ValueError:
    print("Введённые данные не являются числом")



from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
q = (np.matmul(np.linalg.inv(A), np.transpose(A)) - k * F)
h = ((A + np.linalg.inv(np.tril(A)) - np.linalg.inv(F)) * k)
fig = plt.figure()
ax = fig.add_subplot()
if np.linalg.det(A) > sum(np.diagonal(F)) and np.linalg.det(A) != 0:                                                    # примеры для q
    plt.title('plot')                                                                                                   # 1 пример matplotlib
    plt.xlabel('столбец')
    plt.ylabel('значение')
    for j in range(n):
        plt.plot([i for i in range(n)],q[j][::], marker='o')
    plt.grid()
    plt.show()
    plt.title("Bar")                                                                                                    # 2 пример matplotlib
    plt.xlabel("столбец")
    plt.ylabel("значение")
    for i in range(n):
        plt.bar([i for i in range(n)], q[::-1][i], width=1)
    plt.grid()
    plt.show()
    for j in range(n):                                                                                                  # 3 пример matplotlib
        plt.hist(q[j])
    plt.grid()
    plt.show()
    f, ax = plt.subplots(figsize=(9, 6))                                                                                # 1 пример seaborn
    sns.heatmap(q, annot=True, linewidths=.5, ax=ax)
    plt.show()
    sns.catplot(data=pd.DataFrame(q), kind="violin")                                                                    # 2 пример seaborn
    plt.show()
    p = sns.lineplot(data=pd.DataFrame(np.transpose(q)))                                                                # 3 пример seaborn
    p.set_xlabel("Номер элемента в строчке", fontsize=15)
    p.set_ylabel("Значение", fontsize=15)
    plt.show()
elif np.linalg.det(F) != 0 and np.linalg.det(np.tril(A)) != 0:                                                          # примеры для h
    plt.title("plot")                                                                                                   # 1 пример matplotlib
    plt.xlabel("столбец")
    plt.ylabel("значение")
    for j in range(n):
        plt.plot([i for i in range(n)], h[j][::], marker='o')
    plt.grid()
    plt.show()
    plt.title("Bar")                                                                                                    # 2 пример matplotlib
    plt.xlabel("столбец")
    plt.ylabel("значение")
    for i in range(n):
        plt.bar([i for i in range(n)], h[::-1][i], width=1)
    plt.grid()
    plt.show()
    for j in range(n):                                                                                                  # 3 пример matplotlib
        plt.hist(h[j])
    plt.grid()
    plt.show()
    f, ax = plt.subplots(figsize=(9, 6))                                                                                # 1 пример seaborn
    sns.heatmap(h, annot=True, linewidths=.5, ax=ax)
    plt.show()
    sns.catplot(data=pd.DataFrame(h), kind="violin")                                                                    # 2 пример seaborn
    plt.show()
    p = sns.lineplot(data=pd.DataFrame(np.transpose(h)))                                                                # 3 пример seaborn
    p.set_xlabel("Номер элемента в строчке", fontsize=15)
    p.set_ylabel("Значение", fontsize=15)
    plt.show()




