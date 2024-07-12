from genNN import loadModel
import matplotlib.pyplot as plt
import sys
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score 
from sklearn. metrics import pairwise_distances
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from matplotlib.pyplot import plot, savefig


fld = 'results/071124_run_9'
#n_nn = sys.argv[1]
n_nn = 1

sims = []
labels = []
legend = []
baseStates = []
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

# load neural network
# model = loadModel(fld+'/projector/model_'+n_nn)
model = loadModel(fld+'/projector/model_'+'1')
# projected from 74 flux to 2 dimensions

points = model.predict(sims)


# project base states
#pointsBase = []
#for s in baseStates:
	#pointsBase.append(model.predict(s))

#print(baseStates)

### find distances between points 
#distances = squareform(pdist(points))
#for i in range(distances.shape[0]):
	#distances[i,i] = 1e6
#minDists = np.min(distances, axis=1)

## print distances larger than an arbitrary value to find "outliers
#standAlones = []
#for i in range(minDists.shape[0]):
	#if minDists[i] > 0.4:
		#standAlones.append(i)
		#print(uniqueLabels[i])		

#tsne = TSNE().fit_transform(sims)

palette_colors = ["#2d67a6", "#99c2db", "#3a942e", "#a6dd77"]

#plt.subplot(1,2,1)
sns.scatterplot(x=points[:,0], y=points[:,1], hue=labels, palette=palette_colors)

colors = [(0.0, 0.0, 0.0),
		  (1.0, 0.0, 0.0),
		  (0.0, 1.0, 0.0),
		  (0.0, 0.0, 1.0)]

#for i in range(len(baseStates)):
	#p = pointsBase[i]
	#plt.plot(p[:,0], p[:,1], 'o', color=colors[i])

#plt.plot(-2.3809583, -1.7635939, 'r*')
#plt.annotate('w_caf_100_37', xy=(-2.3809583, -1.7635939), xytext=(-2.5 ,-2))
#plt.plot(0.18015312, 1.3038118, 'r*')
#plt.annotate('w_crc_20_53', xy=(0.18015312, 1.3038118))
#plt.plot(0.02325462, 1.9340379, 'r*')
#plt.annotate('w_crc_100_14', xy=(0.02325462, 1.9340379))
#plt.plot(-1.0063322,  2.3427598, 'r*')
#plt.annotate('w_crc_100_42', xy=(-1.0063322,  2.3427598))
#plt.plot(-2.305267 ,  1.6879022, 'r*')
#plt.annotate('k_caf_80_14', xy=(-2.305267 ,  1.6879022))
#plt.plot(-3.0677793 , -0.16940898, 'r*')
#plt.annotate('k_caf_100_66', xy=(-3.0677793 , -0.16940898), xytext=(-3 ,-0.3))
#plt.plot(0.20555137, -1.320513, 'r*')
#plt.annotate('k_crc_100_52', xy=(0.20555137, -1.320513))


#TSNE graph
#plt.subplot(1,2,2)
# custom palette to match original figures
#sns.scatterplot(x=tsne[:,0], y=tsne[:,1], hue=labels, palette=palette_colors)

plt.show()


