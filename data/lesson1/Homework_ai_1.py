import numpy as np


arr = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

# 1. Число 14 (рядок 3, стовпчик 1)
print(arr[3, 1])

# 2. Третій рядок (індекс 2)
print(arr[2])

# 3. Перший стовпчик (всі рядки, стовпчик 0)
print(arr[:, 0])

# 4. Верхня половина (рядки 0 та 1)
print(arr[:2, :])

# 5. Заміна чисел в рядках 2-3 (індекси 1 та 2) на 100
arr[1:3, :] = 100
print(arr)

# 6. Другий рядок стає таким, як останній (індекс 1 = індекс 3)
arr[1] = arr[3]
print(arr)


# 1. Створення маски для парних чисел
mask = (arr % 2 == 0)

# 2. Виведення парних чисел
print(arr[mask])

# 3. Заміна всіх парних чисел на 100
arr[mask] = 100

print(arr)


# Створення масивів типу uint8
array1 = np.array([128, 200, 10], dtype=np.uint8)
array2 = np.array([250, 10, 34], dtype=np.uint8)

# Розрахунок: (0.2 * array1) + (0.8 * array2)
result = (array1.astype(np.float32) * 0.2 + array2.astype(np.float32) * 0.8).astype(np.uint8)

print(array1)
print(array2)
print(result)
print(result.dtype)