import numpy as np
import json
import matplotlib.pyplot as plt
from VSS_Sim import VolterraSteinSteinSimulator
from VSS_COS import VSSParam

param = VSSParam(0.00842271,  0.12275754,  0.84508075, -0.02704256,  0.16772171,  0.09164193)
with open('Data/250901.json', 'r', encoding='utf-8') as f:
    optiondict = json.load(f)

S0 = optiondict['HSI']
r = optiondict['rf']
T = 1.0
n_points = 252

simulator = VolterraSteinSteinSimulator(params=param)
t_grid, X_paths, S_paths = simulator.simulate_vss_path(T, S0, n_points, N_paths=50)

fig, axs = plt.subplots(2, 1, figsize=(10, 8))

for i in range(50):
    axs[0].plot(t_grid, S_paths[i], alpha=0.7)
    axs[0].set_title('Simulated Stock Price Paths')
    axs[0].set_xlabel('Time (years)')
    axs[0].set_ylabel('Stock Price')
    axs[0].grid()

    axs[1].plot(t_grid, X_paths[i], alpha=0.7)
    axs[1].set_title('Simulated Variance Process Paths')
    axs[1].set_xlabel('Time (years)')
    axs[1].set_ylabel('Variance')
    axs[1].grid()
plt.tight_layout()
plt.show()