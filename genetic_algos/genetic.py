import numpy as np
import random
import string

target_string = 'JoshuaPeterEbenezer15EC10023'
string_len = len(target_string)
fitness_threshold = string_len
p = 100
r = 0.5
m = 0.1

def fitness(candidate):
	fitness = 1e-3
	for i in range(string_len):
		if (candidate[i]==target_string[i]):
			fitness+=1
	return fitness

def find_max_fitness(list_of_strings):
	max_fitness = 0
	max_index = 0
	for i,string in enumerate(list_of_strings):
		curr_fit = fitness(string)
		if (curr_fit>max_fitness):
			max_fitness = curr_fit 
			max_index = i
	return max_index,max_fitness

def crossover(parents,offspring_size):
	crossover_point1 = random.randint(0,string_len)
	crossover_point2 = random.randint(crossover_point1+1,string_len+1)
	offspring = []
	offspring_pair_no = int(offspring_size/2)
	for k in range(offspring_pair_no):
		parent1 = parents[2*k]
		parent2 = parents[2*k+1]

		offspring1 = parent1[:crossover_point1]+parent2[crossover_point1:crossover_point2]+parent1[crossover_point2:]
		offspring2 = parent2[:crossover_point1]+parent1[crossover_point1:crossover_point2]+parent2[crossover_point2:]

		offspring.append(offspring1)
		offspring.append(offspring2)
	return offspring

def mutate(random_Ps,choice_string):
	mutate_Ps = []
	for random_string in random_Ps:
		random_position = random.choice(range(string_len))
		# print(random_position)
		random_character = random.choice(choice_string)
		new_random_string = random_string[:random_position] + random_character + random_string[random_position+1:]
		mutate_Ps.append(new_random_string) 
	return mutate_Ps

def main():

	choice_string = string.ascii_letters

	choice_string = choice_string+string.digits
	print(choice_string)
	list_of_strings = []
	prob_denom = 0
	prob_num = np.zeros((p,))

	#choose initial generation randomly
	for i in range(p):
		random_string = ''.join(random.choices(choice_string,k=string_len))

		# print(len(random_string)) 
		list_of_strings.append(random_string)
		curr_fit = fitness(random_string)
		prob_num[i] = curr_fit
		prob_denom += curr_fit
	array_of_probs = prob_num/prob_denom
	max_index,max_fitness = find_max_fitness(list_of_strings)
	print('Fitness_threshold is',fitness_threshold)
	c=0
	P = list_of_strings
	f= open("result.txt","w+")
	while (max_fitness<fitness_threshold):
		c+=1
		select_size = int((1-r)*p)
		Ps = np.random.choice(P,size=select_size,replace=False,p=array_of_probs)	# 1. select

		Ps = list(Ps)
		crossover_size = int(r*p)
		crossover_parents = np.random.choice(P,size=crossover_size,replace=False,p=array_of_probs) # 2. crossover
		offspring = crossover(crossover_parents,crossover_size)

		Ps = Ps+offspring
		random_Ps = [Ps.pop(random.randrange(len(Ps))) for _ in range(int(m*100))]
		mutate_Ps = mutate(random_Ps,choice_string) 												# 3. mutate
	
		Ps = Ps+mutate_Ps

		max_index=0
		max_fitness=0
		prob_denom = 0
		for i,random_string in enumerate(Ps):		#update probabilities 
			curr_fit = fitness(random_string)
			if (curr_fit>max_fitness):
				max_fitness = curr_fit
				max_index = i
			prob_num[i] = curr_fit
			prob_denom += curr_fit
		array_of_probs = prob_num/prob_denom


		print('Iteration:',c,' Max fitness:',max_fitness,'Closest string:',Ps[max_index])
		# line = 
		f.write('Iteration: %d  Max fitness: %d Closest string: %s\n' %(c,max_fitness,Ps[max_index]))
		if (c>10000):
			break;
		P = Ps              	# update generation



if __name__ == "__main__":
	main()
