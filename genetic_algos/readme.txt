Fitness function was defined as number of characters that are the same as the target string
Minimum fitness value was 1e-3 (not 0) so that there would be a non-zero probability for at least few strings (if all are 0 numpy throws an error)

External functions used: random.choice, random.sample

Functions written: crossover, mutate, fitness

Procedure:
Selected ((1-r)*p) of the population from P to put in Ps
Did crossover for r*p/2 parents from P and kept the children in Ps
Mutation for m % of the strings in Ps
Update probabilities and check max fitness
Update P <- Ps
Repeat