import numpy as np
import matplotlib.pyplot as plt

# Grid parameters
grid_size = 9
half = grid_size // 2
x = np.linspace(-half, half, grid_size)
y = np.linspace(-half, half, grid_size)
X, Y = np.meshgrid(x, y)

# Convert to polar coordinates
R = np.sqrt(X**2 + Y**2)
Phi = np.arctan2(Y, X)

def tau_j(r, j, sigma=1.0):
    mu = j
    return np.exp(-((r - mu) ** 2) / (2 * sigma ** 2))

def circular_harmonic(j, k, R, Phi):
    radial = tau_j(R, j)
    angular = np.exp(1j * k * Phi)
    return radial * angular


js = [3, 2, 1, 0]
ks = [0, 1, 2, 3, 4]

fig, axes = plt.subplots(len(js), len(ks) * 2, figsize=(12, 6))
for i, j in enumerate(js):
    for l, k in enumerate(ks):
        psi = circular_harmonic(j, k, R, Phi)
        real = np.real(psi)
        imag = np.imag(psi)

        axes[i, 2 * l].imshow(real, cmap='gray')
        axes[i, 2 * l].set_title(f"Re j={j}, k={k}")
        axes[i, 2 * l].axis('off')

        axes[i, 2 * l + 1].imshow(imag, cmap='gray')
        axes[i, 2 * l + 1].set_title(f"Im j={j}, k={k}")
        axes[i, 2 * l + 1].axis('off')

plt.tight_layout()
plt.show()
