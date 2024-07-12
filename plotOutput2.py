import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from genNN import loadModel

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