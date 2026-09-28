import csv
import matplotlib.pyplot as plt
import numpy as np

data = {}
# parse our data from the csv file
with open('results.csv') as f:
    reader = csv.DictReader(f)
    for row in reader:
        key = f"{row['Mode']}_{row['Type']}"
        data.setdefault(key, []).append(float(row['Time_ms']))

# grab means (although I ended up just printing one value so probably dont need this)
keys = sorted(data.keys())
means = [np.mean(data[k]) for k in keys]
stds = [np.std(data[k]) for k in keys]


# print the chart
plt.figure(figsize=(10, 6))
plt.bar(keys, means, yerr=stds, capsize=5, color=['blue', 'red', 'yellow', 'green'])
plt.title('Performance Comparison: Branching vs No Branching (GPU vs CPU)')
plt.ylabel('Average Time (ms)')
plt.xlabel('Kernel Setup')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('chart.png', dpi=150)
plt.show()
print("CREATED CHART")