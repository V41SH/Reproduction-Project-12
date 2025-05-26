import numpy as np
import matplotlib.pyplot as plt

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

fig, axes = plt.subplots(len(js), len(ks)*2, figsize=(12,6))

for i, j in enumerate(js):
    for l, k in enumerate(ks):
        if j == 0 and k != 0:
            axes[i, 2*l    ].axis('off')
            axes[i, 2*l + 1].axis('off')
            continue

        psi        = circular_harmonic(j, k, R, Phi)
        real       = np.real(psi)
        imag_inv   = np.imag(np.conj(psi))    
        real_mapped = (real     + 1.0) / 2.0  
        imag_mapped = (imag_inv + 1.0) / 2.0

        ax_r = axes[i, 2*l]
        ax_i = axes[i, 2*l + 1]

        ax_r.imshow(real_mapped, cmap='gray', vmin=0, vmax=1)
        ax_r.set_title(f"Re j={j}, k={k}")
        ax_r.axis('off')

        if k != 0:
            ax_i.imshow(imag_mapped, cmap='gray', vmin=0, vmax=1)
            ax_i.set_title(f"Im j={j}, k={k}")
            ax_i.axis('off')
        else:
            ax_i.axis('off')

plt.tight_layout()
plt.show()

# save_path = "circular_harmonics.png"
# fig.savefig(save_path, dpi=300)
# print(f"Figure saved to {save_path}")
