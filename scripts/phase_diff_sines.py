import matplotlib.pyplot as plt
import numpy as np

# Create an array of t-values for plotting the sine waves
t = np.linspace(0, 2 * np.pi, 100)  # t-values from 0 to 2π (full sine wave period)

# Create a figure with 2x2 subplots
fig, axs = plt.subplots(4, 1, figsize=(10, 8))  # 2x2 grid of subplots

# Plot the first subplot with waves in-phase
axs[0].plot(t, np.sin(t), label="sin(t)", color="blue")
axs[0].plot(t, np.sin(t) + 1, label="sin(t)+1", color="red")
axs[0].set_title("In-Phase")
axs[0].legend()

# Plot the second subplot with waves out of phase by π/2 radians
axs[1].plot(t, np.sin(t), label="sin(t)", color="blue")
axs[1].plot(t, np.sin(t + np.pi / 2), label="sin(t + π/2)", color="red")
axs[1].set_title("Out of Phase by π/2")
axs[1].legend()

# Plot the third subplot with waves in anti-phase (out of phase by π radians)
axs[2].plot(t, np.sin(t), label="sin(t)", color="blue")
axs[2].plot(t, np.sin(t + np.pi), label="sin(t + π)", color="red")
axs[2].set_title("Anti-Phase")
axs[2].legend()

# Plot the fourth subplot with waves out of phase by -π/2 radians
axs[3].plot(t, np.sin(t), label="sin(t)", color="blue")
axs[3].plot(t, np.sin(t - np.pi / 2), label="sin(t - π/2)", color="red")
axs[3].set_title("Out of Phase by -π/2")
axs[3].legend()

# Adjust spacing between subplots for better visibility
plt.tight_layout(rect=[0, 0, 1, 0.96])  # Keep some space for the supertitle

# Show the plot
plt.show()
