# import numpy as np

# N = 1000000
# x = np.random.rand(N)
# y = np.random.rand(N)
# inside_circle = (x**2 + y**2) < 1
# pi_estimate = 4 * np.mean(inside_circle)
# print(pi_estimate)

# import torch
# import torch.nn as nn

# m = nn.Softmax(dim=1)
# input = torch.randn(2, 3)
# output = m(input)
# print(output)

# import matplotlib.pyplot as plt
# from matplotlib.widgets import Slider
# import numpy as np

# # Sample data
# x = np.linspace(0, 10, 1000)
# a0 = 5
# y = a0 * np.sin(x)

# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.25)

# line, = ax.plot(x, y)
# ax.set_title('Interactive Sine Wave')

# # Slider axis
# ax_slider = plt.axes([0.2, 0.1, 0.65, 0.03])  # [left, bottom, width, height]
# slider = Slider(ax_slider, 'Amplitude', 0.1, 10.0, valinit=a0)

# # Update function
# def update(val):
#     amp = slider.val
#     line.set_ydata(amp * np.sin(x))
#     fig.canvas.draw_idle()

# slider.on_changed(update)

# plt.show()
