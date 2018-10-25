import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

train_data = pd.read_csv("iris.csv") 
test_data = np.array([7.2, 3.6, 5.1, 2.5])

train_features = train_data[['SepalLength','SepalWidth','PetalLength','PetalWidth']]

distance = np.linalg.norm(train_features.values-test_data,axis=1)	# find the distance of test points from other points
labelled_distance = np.concatenate((np.expand_dims(distance,axis=1),np.expand_dims(train_data['Name'],axis=1)),axis=1)	# create array with corresponding labels
sorted_distances = labelled_distance[labelled_distance[:,0].argsort()]	# sort according to distance from test point

for k in range(1,6):
	c = np.zeros((3,))
	for i in range(k):
		if(sorted_distances[i][1] == 'Iris-setosa'):
			c[0]+=1
		elif(sorted_distances[i][1] == 'Iris-setosa'):
			c[1]+=1
		else:
			c[2]+=1
	if(np.argmax(c)==0):
		test_result = 'Iris-setosa'
	elif(np.argmax(c)==1):
		test_result = 'Iris-versicolor'
	else:
		test_result = 'Iris-virginica'

	print('Prediction by k-NN for k=',k,'by handwritten KNeighborsClassifier is', test_result)
	neigh = KNeighborsClassifier(n_neighbors=k)
	neigh.fit(train_features, train_data['Name'])
	scikit_result = neigh.predict(test_data.reshape(1,-1))
	print('Prediction by k-NN for k=',k,'by scikit KNeighborsClassifier is',scikit_result)
	print()