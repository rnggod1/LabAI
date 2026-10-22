import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import r2_score, mean_squared_error


# 1. Загрузка данных

data = pd.read_csv('ticdata2000.txt', sep=r'\s+', header=None)

features = data.iloc[:, :-1]
target = data.iloc[:, -1]

print("Количество строк:", len(data))
print("Количество признаков:", features.shape[1])


# 2. Разделение на обучающую и тестовую выборки
# Без train_test_split

np.random.seed(42)

indices = np.random.permutation(len(data))
train_size = int(len(data) * 0.8)

train_indices = indices[:train_size]
test_indices = indices[train_size:]

x_train = features.iloc[train_indices]
x_test = features.iloc[test_indices]

y_train = target.iloc[train_indices]
y_test = target.iloc[test_indices]

print("Обучающая выборка:", len(x_train))
print("Тестовая выборка:", len(x_test))


# Масштабирование признаков

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


# 3. Линейная регрессия

linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

train_prediction = linear_model.predict(x_train)
test_prediction = linear_model.predict(x_test)

print("\nЛинейная регрессия")
print("R2 обучение:", r2_score(y_train, train_prediction))
print("R2 тест:", r2_score(y_test, test_prediction))
print("MSE тест:", mean_squared_error(y_test, test_prediction))


# 4. Полиномиальная регрессия

# Используем степени отдельных признаков без
# попарных взаимодействий, чтобы избежать
# слишком большого количества столбцов.

degrees = [1, 2, 3, 4, 5]

train_scores = []
test_scores = []

for degree in degrees:
    polynomial = PolynomialFeatures(
        degree=degree,
        include_bias=False,
        interaction_only=False
    )

    # Для контроля размера модели оставляем
    # только степени каждого признака.
    train_parts = [x_train ** power
                   for power in range(1, degree + 1)]
    test_parts = [x_test ** power
                  for power in range(1, degree + 1)]

    x_train_poly = np.hstack(train_parts)
    x_test_poly = np.hstack(test_parts)

    # Масштабирование полиномиальных признаков
    poly_scaler = StandardScaler()
    x_train_poly = poly_scaler.fit_transform(x_train_poly)
    x_test_poly = poly_scaler.transform(x_test_poly)

    polynomial_model = LinearRegression()
    polynomial_model.fit(x_train_poly, y_train)

    train_prediction = polynomial_model.predict(x_train_poly)
    test_prediction = polynomial_model.predict(x_test_poly)

    train_score = r2_score(y_train, train_prediction)
    test_score = r2_score(y_test, test_prediction)

    train_scores.append(train_score)
    test_scores.append(test_score)

    print(f"\nСтепень {degree}")
    print("R2 обучение:", train_score)
    print("R2 тест:", test_score)


# График зависимости R2 от степени полинома

plt.figure(figsize=(9, 5))

plt.plot(degrees, train_scores, 'bo-', label='Обучение')
plt.plot(degrees, test_scores, 'ro-', label='Тест')

plt.title('Точность полиномиальной регрессии')
plt.xlabel('Степень полинома')
plt.ylabel('R2')
plt.xticks(degrees)
plt.legend()
plt.grid(True)
plt.show()


# 5. Регуляризация Ridge

# Используем полиномиальные признаки степени 2

x_train_poly = np.hstack([x_train, x_train ** 2])
x_test_poly = np.hstack([x_test, x_test ** 2])

poly_scaler = StandardScaler()
x_train_poly = poly_scaler.fit_transform(x_train_poly)
x_test_poly = poly_scaler.transform(x_test_poly)

coefficients = np.logspace(-3, 5, 20)

ridge_train_scores = []
ridge_test_scores = []

for coefficient in coefficients:
    ridge_model = Ridge(alpha=coefficient)
    ridge_model.fit(x_train_poly, y_train)

    train_prediction = ridge_model.predict(x_train_poly)
    test_prediction = ridge_model.predict(x_test_poly)

    train_score = r2_score(y_train, train_prediction)
    test_score = r2_score(y_test, test_prediction)

    ridge_train_scores.append(train_score)
    ridge_test_scores.append(test_score)


# График зависимости R2 от коэффициента регуляризации

plt.figure(figsize=(9, 5))

plt.semilogx(
    coefficients, ridge_train_scores,
    'bo-', label='Обучение'
)

plt.semilogx(
    coefficients, ridge_test_scores,
    'ro-', label='Тест'
)

plt.title('Влияние коэффициента регуляризации')
plt.xlabel('Коэффициент регуляризации alpha')
plt.ylabel('R2')
plt.legend()
plt.grid(True)
plt.show()


# 6. Подбор лучшего коэффициента

best_index = np.argmax(ridge_test_scores)
best_alpha = coefficients[best_index]

print("\nЛучший коэффициент регуляризации:", best_alpha)
print("Лучший R2 на тесте:", ridge_test_scores[best_index])