import matplotlib.pyplot as plt
import numpy as np

# Set up the figure with equal aspect ratio to avoid distortions
plt.figure(figsize=(6, 6))
plt.axis("equal")  # Ensures the unit circle maintains its shape

# Create a unit circle
theta = np.linspace(0, 2 * np.pi, 100)  # Generate angles for a full circle
x = np.cos(theta)  # x-coordinates
y = np.sin(theta)  # y-coordinates
plt.plot(x, y, "k")  # Plot the unit circle

# Draw x-axis and y-axis with dashed lines
plt.axhline(0, color="black", linewidth=0.5, linestyle="--")  # x-axis
plt.axvline(0, color="black", linewidth=0.5, linestyle="--")  # y-axis


plt.xlim(-1.2, 1.2)
plt.ylim(-1.2, 1.2)

# * Label phase in radians
plt.text(1.1, 0, "0", fontsize=18, color="k")
plt.text(0, 1.1, rf"$\pi/2$", fontsize=18, color="k")
plt.text(-1.1, 0, rf"$\pi$", fontsize=18, color="k")
plt.text(0, -1.1, rf"$-\pi/2$", fontsize=18, color="k")

# * Label regions of lead/lag, phase/anti-phase relationships
plt.text(0.5, 0.5, rf"$x$ leads $y$", fontsize=14, color="k")
plt.text(-0.75, 0.5, rf"$y$ leads $x$ (anti-phase)", fontsize=14, color="k")
plt.text(-0.75, -0.5, rf"$x$ leads $y$ (anti-phase)", fontsize=14, color="k")
plt.text(0.5, -0.5, rf"$y$ leads $x$", fontsize=14, color="k")

# * Add example of phase difference arrow
angle = np.pi / 3
x_arrow = np.cos(angle)
y_arrow = np.sin(angle)

plt.arrow(
    0,
    0,
    x_arrow - 0.07,
    y_arrow - 0.07,
    head_width=0.1,
    head_length=0.1,
    fc="k",
    ec="k",
    linestyle="-",
    linewidth=1,
    label=rf"$\phi_{{xy}}$",
)

# * Strip axis labels
plt.gca().set_xticks([])
plt.gca().set_yticks([])

# * Remove frame
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().spines["bottom"].set_visible(False)
plt.gca().spines["left"].set_visible(False)

# * Include legend
plt.legend()

# Show the plot
plt.show()
