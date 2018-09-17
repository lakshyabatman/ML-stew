import numpy as np
from sympy.utilities.iterables import multiset_permutations

layer_no = 3

hidden_layer_nodes = 5
output_nodes = 3 

def outer(outputs,targets):
	# delta E/ delta wji = delta E/delta netj * delta netj/delta wji
	# delta E/delta netj = delta E/delta oj * delta oj/ delta netj
	oj = outputs
	tj = targets
	delta_oj_by_delta_netj = oj*(1-oj)
	delta_E_by_delta_oj = (tj-oj)*-1
	delta_E_by_delta_netj = np.sum(delta_E_by_delta_oj*delta_oj_by_delta_netj)


	return delta_E_by_delta_netj

def inner(output,delta_k,wkj):
	# delta_E_by_delta_netj = sigma ( delta_E_by_delta_netk * delta_netk_by_delta_netj )
	# delta_netk_by_delta_netj = delta_netk_by_oj * delta_oj_by_netj
	# delta_oj_by_netj = oj(1-oj)
	# delta_netk_by_oj = wkj
	# delta_E_by_delta_netk = delta_k

	oj = output
	delta_oj_by_netj = oj*(1-oj)
	delta_E_by_delta_netj = np.sum(delta_k*wkj)*delta_oj_by_netj

	return delta_E_by_delta_netj


def sigmoid(val):
	sigm = 1/(1+np.exp(-val))
	return sigm

def fwd_node(inputs,weights):
	out= sigmoid(np.sum(weights * inputs))
	return out

def forward(inputs,weights_1,weights_2):
	out_hidden = np.zeros(hidden_layer_nodes)
	out = np.zeros(output_nodes)

	for j in range(hidden_layer_nodes):
		out_hidden[j] = fwd_node(inputs,weights_1[j])
	out_hidden = np.insert(out_hidden,0,1)

	for j in range(output_nodes):	
		out[j] = fwd_node(out_hidden,weights_2[j,:]) 
	return out,out_hidden

def backward(outputs,targets,weights_2,out_hidden,inputs):
	eta = 0.8
#oh man this is not easy

# get updates for outer nodes
	outer_wji_delta = np.empty((outputs.shape[0]))
	outer_wji_update= np.empty((outputs.shape[0],out_hidden.shape[0]))
	for j in range(outputs.shape[0]):

		outer_wji_delta[j] = outer(outputs[j],targets[j])

		for i in range(out_hidden.shape[0]):
			xji = out_hidden[i]
			outer_wji_update[j,i] = -eta * xji * outer_wji_delta[j]		
	

# get updates for inner nodes
	hidden_wji_delta = np.empty((out_hidden.shape[0]-1))
	hidden_wji_update= np.empty((out_hidden.shape[0]-1,inputs.shape[0]))

	for j in range(out_hidden.shape[0]-1):
		weightsj = weights_2[:,j+1]
		hidden_wji_delta[j] = inner(out_hidden[j],outer_wji_delta,weightsj)
		for i in range(inputs.shape[0]):
			xji = inputs[i]
			hidden_wji_update[j,i] = -eta * xji * hidden_wji_delta[j]
	return hidden_wji_update,outer_wji_update
	

a = np.array((1,0,0,0,0,0,0,0)) #,01000000,00100000,00010000,00001000,00000100,00000010,00000001])

inputs = np.empty((8,9))

for i,p in enumerate(multiset_permutations(a)):
	inputs[i,1:9] = p
	inputs[i,0] = 1

targets = np.array([[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]])
weights_1 = -1+2*np.random.rand(5,9)
weights_2 = -1+2*np.random.rand(3,6)

# print(weights_2[:,0])

for epoch in range(100):
	error = 0
	epoch_hidden_update = 0
	epoch_outer_update = 0
	for i,example in enumerate(inputs):
		out,out_hidden = forward(example,weights_1,weights_2)
		error+=np.square(targets[i]-out)
		hidden_wji_update,outer_wji_update = backward(out,targets[i],weights_2,out_hidden,example)
		
		epoch_hidden_update+=hidden_wji_update
		epoch_outer_update += outer_wji_update

	weights_1 = weights_1+epoch_hidden_update
	weights_2 = weights_2+epoch_outer_update
	print('Error in epoch ', epoch, 'is ', np.average(error))

for i,example in enumerate(inputs):
	out,out_hidden = forward(example,weights_1,weights_2)
	print(example,targets[i],out)
print('Hidden weights ',weights_1)
print('Outer weights ',weights_2)




