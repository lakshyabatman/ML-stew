import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


def compute_labels(xy_coord,centroid,k):
	label = np.zeros(shape=(xy_coord.shape[0],),dtype=np.int32)
	for i in range(xy_coord.shape[0]):	#iterate over all coordinates
		min_dist = np.linalg.norm(xy_coord[i,:]-centroid[0,:])
		label[i] = 0
		for j in range(1,k):				#iterate over all centroids
			new_dist = np.linalg.norm(xy_coord[i,:]-centroid[j,:])
			if(new_dist<min_dist):				# find nearest centroid
					min_dist = new_dist
					label[i] = j
	return label

def kmeans(xy_coord,k):
	centroid = np.zeros(shape=(k,2))
	mean = np.floor(np.average(xy_coord,axis=0))
	std_dev = np.floor(np.std(xy_coord,axis=0))

	for i in range(k):
  																	# random initialization of centroids
		centroid[i,0] = random.randint(mean[0]-std_dev[0],mean[0]+std_dev[0])
		centroid[i,1] = random.randint(mean[1]-std_dev[1],mean[1]+std_dev[1])
  	
  	# initial estimation
	label = compute_labels(xy_coord,centroid,k)

	for iteration in range(10):

		cluster_sum = np.zeros(shape=(k,2))
		cluster_mean = np.zeros(shape=(k,2))
		cluster_count = np.zeros(shape=(k,))
		for j in range(k):
			for i in range(xy_coord.shape[0]):
				if (label[i]==j):
					cluster_sum[j,:]+=xy_coord[i,:]
					cluster_count[j]+=1
			cluster_mean[j,:] = cluster_sum[j,:]/cluster_count[j]	#compute new centroid of that cluster		
		centroid = cluster_mean
		label = compute_labels(xy_coord,centroid,k)

	return (centroid,label)



xy_coord = pd.read_excel(open('dataset.xlsx','rb'))

xy_coord = xy_coord.values
avg_error = np.zeros(shape=(5,))

for k in range(1,6):
	centroid,label = kmeans(xy_coord,k) #scipy.cluster.vq.kmeans2(xy_coord, k)
	error = np.linalg.norm(xy_coord-centroid[label], ord=2,axis=1)
	avg_error[k-1] = np.average(error)
	print("Average error for k = ",k," is ", avg_error[k-1])
np.savetxt("result.txt",avg_error)
plt.plot(range(1,6),avg_error)
plt.xlabel('k (number of clusters)')
plt.ylabel('error')
plt.title('Error vs number of clusters')
plt.show()
