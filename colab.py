from glob import glob

import numpy
import random


class geneticAlgorithmForQAP:

    def __init__(self, populationSize, mutationRate, numberOfIterations, dimensions, weightMatrix, distanceMatrix):
        self._populationSize = populationSize
        self._mutationRate = mutationRate
        self._numberOfIterations = numberOfIterations
        self._dimensions = dimensions
        self._weight_matrix = weightMatrix
        self._distance_matrix = distanceMatrix

    def initializePopulation(self):  # Initialize population, done in a more pythonic way than the paper.
        population = numpy.zeros(shape=(self._populationSize, self._dimensions), dtype='uint8')
        for i in range(self._populationSize):
            randomNonRepeatingList = [num for num in range(self._dimensions)]
            random.shuffle(randomNonRepeatingList)
            for j in range(self._dimensions):
                population[i][j] = randomNonRepeatingList[j]
        return population

    def dealWithUnsuccessfulSwap(self, unssuccessfulSwapsParentOne, unssuccessfulSwapsParentTwo, childOne,
                                 childTwo):  # Used to make chromosome valid
        while (len(unssuccessfulSwapsParentOne) * len(unssuccessfulSwapsParentTwo) != 0):
            parentForHorizontalSwap = random.randint(1, 2)
            elementOne = unssuccessfulSwapsParentOne.pop(random.randint(0, len(unssuccessfulSwapsParentOne) - 1))
            elementTwo = unssuccessfulSwapsParentTwo.pop(random.randint(0, len(unssuccessfulSwapsParentTwo) - 1))
            if (parentForHorizontalSwap == 1):
                childOne[elementOne[1]] = childOne[elementTwo[1]]
                childOne[elementTwo[1]] = elementTwo[0]
                childTwo[elementTwo[1]] = elementOne[0]
            else:
                childTwo[elementTwo[1]] = childTwo[elementOne[1]]
                childTwo[elementOne[1]] = elementOne[0]
                childOne[elementOne[1]] = elementTwo[0]
        elements = set()
        for i in range(0, self._dimensions):
            elements.add(i)
        if (len(unssuccessfulSwapsParentOne) > 0):
            for j in range(len(childTwo)):
                if (childTwo[j] in elements):
                    elements.remove(childTwo[j])
                else:
                    elementOne = unssuccessfulSwapsParentOne.pop(
                        random.randint(0, len(unssuccessfulSwapsParentOne) - 1))
                    childOne[elementOne[1]] = childTwo[j]
                    childTwo[j] = elementOne[0]
        elif (len(unssuccessfulSwapsParentTwo) > 0):
            for j in range(len(childOne)):
                if (childOne[j] in elements):
                    elements.remove(childOne[j])
                else:
                    elementTwo = unssuccessfulSwapsParentTwo.pop(
                        random.randint(0, len(unssuccessfulSwapsParentTwo) - 1))
                    childTwo[elementTwo[1]] = childOne[j]
                    childOne[j] = elementTwo[0]

        return (childOne, childTwo)

    def crossover(self, parentOne, parentTwo):
        pointOne = random.randint(0, len(parentOne))
        pointTwo = random.randint(pointOne, len(parentTwo))
        parentOneFixed = set()
        parentTwoFixed = set()
        for i in range(pointOne, pointTwo):
            parentOneFixed.add(parentOne[i])
            parentTwoFixed.add(parentTwo[i])
        childOne = parentOne.copy()
        childTwo = parentTwo.copy()
        unssuccessfulSwapsParentOne = []
        unssuccessfulSwapsParentTwo = []
        for j in range(0, pointOne):
            b1 = parentOne[j] not in parentTwoFixed
            b2 = parentTwo[j] not in parentOneFixed
            if (b1 and b2):
                childOne[j] = parentTwo[j]
                childTwo[j] = parentOne[j]
            elif (b1):
                unssuccessfulSwapsParentOne.append((parentOne[j], j))
            elif (b2):
                unssuccessfulSwapsParentTwo.append((parentTwo[j], j))
        for k in range(pointTwo, len(parentOne)):
            b1 = parentOne[k] not in parentTwoFixed
            b2 = parentTwo[k] not in parentOneFixed
            if (b1 and b2):
                childOne[k] = parentTwo[k]
                childTwo[k] = parentOne[k]
            elif (b1):
                unssuccessfulSwapsParentOne.append((parentOne[k], k))
            elif (b2):
                unssuccessfulSwapsParentTwo.append((parentTwo[k], k))
        childOne, childTwo = self.dealWithUnsuccessfulSwap(unssuccessfulSwapsParentOne, unssuccessfulSwapsParentTwo,
                                                           childOne, childTwo)
        return childOne, childTwo

    def mutate(self, chromosome):
        indices = [x for x in range(self._dimensions)]
        indices = random.sample(indices, 2)
        chromosome[indices[0]] = chromosome[indices[0]] + chromosome[indices[1]]
        chromosome[indices[1]] = chromosome[indices[0]] - chromosome[indices[1]]
        chromosome[indices[0]] = chromosome[indices[0]] - chromosome[indices[1]]
        return chromosome

    def QAP(self, chromosome):  # QAP objective function
        toReturn = 0
        for row in range(len(self._distance_matrix)):
            for col in range(len(self._distance_matrix[row])):
                toReturn += self._weight_matrix[chromosome[row]][chromosome[col]] * self._distance_matrix[row][col]
        return toReturn

    def indicesFromRouletteWheelSelection(self, population):
        chromosomeIndexAndFitness = [(i, 1 / self.QAP(population[i])) for i in range(len(population))]
        bounds = []
        prev = 0
        next = 0
        for i in range(len(chromosomeIndexAndFitness)):
            next = prev + chromosomeIndexAndFitness[i][1]
            bounds.append((prev, next))
            prev = next
        indices = [random.random() * next for j in range(len(population))]
        for j in range(len(indices)):
            k = 0
            while (k < len(bounds)):
                if (indices[j] >= bounds[k][0] and indices[j] <= bounds[k][1]):
                    indices[j] = chromosomeIndexAndFitness[k][0]
                    k = len(bounds)
                k += 1
        return indices

    def runGeneticAlgorithm(self, filename, iteration):
        population = self.initializePopulation()
        bestSolution = population[0].copy()
        bestSolutionFitness = self.QAP(population[0])

        for t in range(self._numberOfIterations):

            if (t % 200 == 0):
                #f.write("Iteration: " + str(t) + " Best solution fitness: " + str(bestSolutionFitness) +  " Best solution: " + str(bestSolution) + "\n")
                print("Iteration:", t, "Best solution fitness:", bestSolutionFitness, "Best solution:", bestSolution)

                indices = self.indicesFromRouletteWheelSelection(population)
            children = []
            for i in range(len(indices)):
                randomIndex = indices[random.randint(0, len(indices) - 1)]
                childOne, childTwo = self.crossover(population[indices[i]], population[randomIndex])
                if (random.uniform(0, 1) < self._mutationRate):
                    childOne = self.mutate(childOne)
                if (random.uniform(0, 1) < self._mutationRate):
                    childTwo = self.mutate(childTwo)
                children.append(childOne)
                children.append(childTwo)
                if (self.QAP(childOne) < bestSolutionFitness):
                    bestSolution = childOne.copy()
                    bestSolutionFitness = self.QAP(childOne)
                if (self.QAP(childTwo) < bestSolutionFitness):
                    bestSolution = childTwo.copy()
                    bestSolutionFitness = self.QAP(childTwo)
            for k in range(len(population)):
                population[k] = children[k]

        #f.write("BS: " + str(bestSolution) + "\n")

        print("Best solution:", bestSolution)

      #  f.write("BSF: " + str(bestSolutionFitness) + "\n")
        print("Best solution fitness:", bestSolutionFitness)


'''fileToRead = open("./instances/nug8", 'r')
dim = int(fileToRead.readline())
counter = 0
dist = []
while (counter < dim):
    line = fileToRead.readline().split()
    if (len(line) > 0):
        row = []
        for x in range(len(line)):
            row.append(int(line[x]))
        dist.append(row)
        counter += 1

weight = []
counter = 0
while (counter < dim):
    line = fileToRead.readline().split()
    if (len(line) > 0):
        row = []
        for x in range(len(line)):
            row.append(int(line[x]))
        weight.append(row)
        counter += 1

print(len(dist))
print(len(weight))

'''

#GA = geneticAlgorithmForQAP(100, 0.1, 500, dim, weight, dist)



dim = int(12)

dist = d


weight = w

filename = "nug12"
print(filename)
GA = geneticAlgorithmForQAP(100, 0.1, 1000, dim, weight, dist)

GA.runGeneticAlgorithm(filename, i)
