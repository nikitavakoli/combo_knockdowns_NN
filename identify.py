import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from genNN import loadModel
from scipy.spatial import KDTree

# Load data and model
fld = 'results/071124_run_9'
n_nn = 1

sims = []
labels = []
uniqueLabels = []

for i in ['KRAS', 'WT']:
    for j in ['CAF', 'CRC']:
        s = np.loadtxt('data/'+i+'_'+j+'.csv', delimiter=',')
        for q in range(s.shape[0]):
            labels.append(i+'_'+j)
            uniqueLabels.append(i+'_'+j+str(q))
        sims.append(s)

sims = np.vstack(sims)
labels = np.array(labels)

# Load neural network
model = loadModel(fld+'/projector/model_'+'1')

# Project data
points = model.predict(sims)

# Create a mapping from points to sims
points_to_sims = {tuple(point): sim for point, sim in zip(points, sims)}

# Define palette colors
palette_colors = {
    'KRAS_CAF': "#3a942e", # dark green
    'KRAS_CRC': "#a6dd77", # light green
    'WT_CAF': "#2d67a6",   # dark blue
    'WT_CRC': "#99c2db"    # light blue
}

# Create a mapping from labels to colors
label_color_map = [palette_colors[label] for label in labels]

# Plot data
plt.figure(figsize=(10, 6))
sns.scatterplot(x=points[:, 0], y=points[:, 1], hue=labels, palette=palette_colors, legend='full')
plt.legend(title='Labels', loc='upper left')
plt.show()

# Build a KDTree for quick nearest neighbor search
kdtree = KDTree(points)

# Define the target point
target_point = np.array([2.308, 0.993])

# Find the nearest point in the projected data
distance, index = kdtree.query(target_point)
nearest_point = points[index]
corresponding_sim = points_to_sims[tuple(nearest_point)]

print("Nearest point in projection:", nearest_point)
print("Corresponding sim:", corresponding_sim)