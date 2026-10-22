import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x1_values = np.linspace(-1, 1, 20)
x2_values = np.linspace(-1, 1, 20)

x1_grid, x2_grid = np.meshgrid(x1_values, x2_values)

y_values = np.tan(x1_grid + x2_grid) * np.sin(x1_grid)

data = pd.DataFrame({
    'x1': x1_grid.flatten(),
    'x2': x2_grid.flatten(),
    'y': y_values.flatten()
})

data.to_csv('data.csv', index=False)

print("Файл data.csv создан")
print("Количество строк:", len(data))


data = pd.read_csv('data.csv')


fixed_x2 = x2_values[10]

graph_x1 = data[np.isclose(data['x2'], fixed_x2)]
graph_x1 = graph_x1.sort_values('x1')

plt.figure(figsize=(8, 5))
plt.plot(graph_x1['x1'], graph_x1['y'], 'bo-')
plt.title(f'y(x1), x2 = {fixed_x2:.3f}')
plt.xlabel('x1')
plt.ylabel('y')
plt.grid(True)
plt.show()


fixed_x1 = x1_values[10]

graph_x2 = data[np.isclose(data['x1'], fixed_x1)]
graph_x2 = graph_x2.sort_values('x2')

plt.figure(figsize=(8, 5))
plt.plot(graph_x2['x2'], graph_x2['y'], 'ro-')
plt.title(f'y(x2), x1 = {fixed_x1:.3f}')
plt.xlabel('x2')
plt.ylabel('y')
plt.grid(True)
plt.show()


for column in data.columns:
    print(f"\nСтолбец {column}:")
    print("Среднее:", data[column].mean())
    print("Минимум:", data[column].min())
    print("Максимум:", data[column].max())


average_x1 = data['x1'].mean()
average_x2 = data['x2'].mean()

filtered_data = data[
    (data['x1'] < average_x1) |
    (data['x2'] < average_x2)
]

filtered_data.to_csv('filtered_data.csv', index=False)

print("\nФайл filtered_data.csv создан")
print("Количество строк:", len(filtered_data))


figure = plt.figure(figsize=(10, 7))
axes = figure.add_subplot(111, projection='3d')

axes.plot_surface(
    x1_grid,
    x2_grid,
    y_values,
    cmap='viridis'
)

axes.set_title('3D график функции')
axes.set_xlabel('x1')
axes.set_ylabel('x2')
axes.set_zlabel('y')

plt.show()