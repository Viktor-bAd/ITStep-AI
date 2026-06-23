# Завдання 1
# Створіть масив з числами від 1 до 10. Виведіть його, його
# розмір, тип даних.
# Змініть розмір масиву на (5, 2). Знову виведіть масив,
# розмір та тип даних
import numpy as np
from numpy.conftest import dtype


# nums = np.arange(1,11)
# print(nums)
# print(nums.shape)
# print(nums.dtype)
#
# new_nums = nums.reshape(2,5)
# print(new_nums)
# print(new_nums.shape)
# print(new_nums.dtype)


# Створіть масив:
# 1 2 3 4
# 5 6 7 8
# 9 10 11 12
# Використовуючи індекси виведіть:
# ● число 7
# ● другий рядок
# ● останній стовпчик
# ● праву половину
# ● жовту область
# ● замініть жовту область на -1
# ● зробіть перший стовпчик таким самим як і другий

nums1 = np.arange(1,13)
new_nums1 = nums1.reshape(3,4)
# print(new_nums1)
# print(new_nums1.shape)
# print(new_nums1.dtype)
#
#
# print(new_nums1[1,2])
# print(new_nums1[1,:])
# print(new_nums1[1:3,1:3])
# print(new_nums1[:,-1])
# print(new_nums1[:,-2:])
# new_nums1[1:3,1:3] = -1
# print(new_nums1)
# new_nums1[:,0] = new_nums1[:,1]
# print(new_nums1)


# Завдання 3
# У масиві з попереднього завдання створіть маску для
# чисел які більші за 6. З її допомогою
# ● виведіть кількість чисел більших за 6
# ● виведіть самі числа
# ● до кожного числа яке відповідає масці додайте 10
# ● кожне число що не відповідає масці помножте на -1
# ● замініть ці числа які відповідають масці на відповідні
# їм з масиву

mask = new_nums1 > 6
print(new_nums1[mask])
count = np.sum(mask)
print(count)

nums_than_6 = new_nums1[mask]
print(nums_than_6)

new_nums1[mask] += 10
print(new_nums1)

new_nums1[~mask] *= -1
print(new_nums1)

nums_than_6 = new_nums1[mask]
print(nums_than_6)


# Завдання 6

array = np.array([10, 4, 25, 40, 200], dtype='uint8')
array = array.astype(np.uint64)
array *= 2
mask = array > 255
array[mask] = 255
print(mask)






