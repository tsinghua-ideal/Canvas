import os
import json
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from scipy.stats import gaussian_kde

# Set the path
folder_path =  "/scorpio/home/shenao/myProject/Canvas/experiments/collections/metrics"  

# Set the data container
data_dict = {}

group_number = 0
print(f'number of sets of data:{len(os.listdir(folder_path))}')
for subfolder in os.listdir(folder_path):
    subfolder_path = os.path.join(folder_path, subfolder)
    if os.path.isdir(subfolder_path):
        for filename in os.listdir(subfolder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(subfolder_path, filename)
                with open(file_path, 'r') as file:
                    json_data = json.load(file)
                    alphas_data = json_data.get('extra').get('magnitude_alphas')  
                    # alphas_data = json_data.get('extra',{}).get('one_hot_alphas')
                    group_number += 1
                    if alphas_data:
                        data_dict[group_number] = alphas_data
                        
# Get the data
x_values = []  # Store the x values (start_epoch)
for i, (filename, alphas_data) in enumerate(data_dict.items()):
    start_epoch = 0
    for j in range(1, len(alphas_data)):
        if torch.argmax(torch.tensor(alphas_data[j])) != torch.argmax(torch.tensor(alphas_data[j - 1])):
            start_epoch = j     
    x_values.append(int(start_epoch))

# Set figure size
plt.figure(figsize=(10, 10))

# Plotting histograms
hist, bins, _ = plt.hist(x_values, alpha=0.6, label=f'Histogram, number of data:{len(os.listdir(folder_path))}', histtype='bar', color='lightblue')

# Plotting kernel density estimation graph
kde = gaussian_kde(x_values)
x_grid = np.linspace(min(x_values), max(x_values), 100)
kde_values = kde(x_grid)
scale_factor = len(x_values) * (bins[1] - bins[0])
kde_values_scaled = kde_values * scale_factor
# plt.plot(x_grid, kde_values_scaled, label='Kernel Density Estimation', color='orange', linewidth=2)

plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Density', fontsize=14)
plt.title('Experiment result of selector')
plt.grid()
plt.tight_layout()  
plt.legend(loc='upper right')
plt.savefig('experiment_result_of_selector.png', dpi=300)  
plt.show()
