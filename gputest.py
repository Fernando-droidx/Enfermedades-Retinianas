import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

print(f"TensorFlow version: {tf.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Is GPU available: {tf.config.list_physical_devices('GPU')}")

# Gráfica de prueba
x = np.linspace(-10, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.title("Test Plot")
plt.xlabel("x")
plt.ylabel("sin(x)")
plt.grid(True)
plt.show()
