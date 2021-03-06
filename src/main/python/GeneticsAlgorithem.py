import random, ProjectsManager, StudentGroup
from types import FunctionType


class Population:
    def __init__(self, distributionManager, populationSize, initialise, students):
        self.students = students
        self.distributions = []
        for i in range(0, populationSize):
            self.distributions.append(None)

        if initialise:
            for i in range(0, populationSize):
                newDistribution = ProjectsManager.Distribution(distributionManager)
                if random.random() < 0.01:
                    newDistribution.generateIndividualUsingCSP(students)
                else:
                    newDistribution.generateIndividual()
                self.saveDistribution(i, newDistribution)

    def __setitem__(self, key, value):
        self.distributions[key] = value

    def __getitem__(self, index):
        return self.distributions[index]

    def saveDistribution(self, index, tour):
        self.distributions[index] = tour

    def getDistribution(self, index):
        return self.distributions[index]

    def getFittest(self):
        fittest = self.distributions[0]
        for i in range(0, self.populationSize()):
            if fittest.getFitness(self.students) <= self.getDistribution(i).getFitness(self.students):
                fittest = self.getDistribution(i)
        return fittest

    def populationSize(self):
        return len(self.distributions)


class GA:
    def __init__(self, distributionManager, students):
        self.students = students
        self.distributionManager = distributionManager
        self.mutationRate = 0.05
        self.tournamentSize = 20
        self.elitism = True
        self.CrossOverMachine = CrossOverMachine(distributionManager,[0.5,0,0.5])

    def evolvePopulation(self, pop):
        newPopulation = Population(self.distributionManager, pop.populationSize(), False, self.students)
        elitismOffset = 0
        if self.elitism:
            newPopulation.saveDistribution(0, pop.getFittest())
            elitismOffset = 1

        for i in range(elitismOffset, newPopulation.populationSize()):
            parent1 = self.tournamentSelection(pop)
            parent2 = self.tournamentSelection(pop)
            while parent2 == parent1:
                parent2 = self.tournamentSelection(pop)

            child = self.CrossOverMachine.calculate(parent1,parent2)
            newPopulation.saveDistribution(i, child)

        for i in range(elitismOffset, newPopulation.populationSize()):
            self.mutate(newPopulation.getDistribution(i))

        return newPopulation

    def mutate(self, distribution):
        for distributionPos1 in range(0, distribution.distributionSize()):
            if random.random() < self.mutationRate:
                distributionPos2 = random.randrange(0, distribution.distributionSize())

                project1 = distribution.getProject(distributionPos1)
                project2 = distribution.getProject(distributionPos2)

                distribution.setProject(distributionPos2, project1)
                distribution.setProject(distributionPos1, project2)

    def tournamentSelection(self, pop):
        tournament = Population(self.distributionManager, self.tournamentSize, False, self.students)
        for i in range(0, self.tournamentSize):
            randomId = random.randrange(0, pop.populationSize())
            tournament.saveDistribution(i, pop.getDistribution(randomId))
        fittest = tournament.getFittest()
        return fittest


class CrossOverMachine:

    def __init__(self,distributionManager, props=None):
        self.distributionManager = distributionManager
        self.props = [0.475, 0.05, 0.475]
        if props is not None:
            if sum(props) != 1:
                raise Exception("Sum of values should be 100%")
            self.dist = props

    def calculate(self, parent1, parent2):
        list = [ y for x, y in CrossOverMachine.__dict__.items() if type(y) == FunctionType and x.startswith('crossover')]
        x=random.random()
        for i in range(0,max(len(list),len(self.props))):
            if x<=self.props[i]:
                return list[i](self,parent1 , parent2)
            x -= self.props[i]
        return list[0] (self,parent1 , parent2)

    def crossoverV1(self, parent1, parent2):
        child = ProjectsManager.Distribution(self.distributionManager)

        startPos = random.randrange(0, parent1.distributionSize())
        endPos = random.randrange(0, parent1.distributionSize())
        if endPos < startPos:
            startPos, endPos = endPos, startPos
        for i in range(0, child.distributionSize()):
            if startPos < i < endPos:
                child.setProject(i, parent1.getProject(i))
        for i in range(0, parent2.distributionSize()):
            if not child.containsProject(parent2.getProject(i)):
                for ii in range(0, child.distributionSize()):
                    if child.getProject(ii) is None:
                        child.setProject(ii, parent2.getProject(i))
                        break
        return child

    def crossoverV2(self, parent1, parent2):
        child = ProjectsManager.Distribution(self.distributionManager)
        ip1 = 0
        ip2 = 0
        for i in range(0, child.distributionSize()):
            while ip1 < parent1.distributionSize() and child.containsProject(parent1.getProject(ip1)):
                ip1 += 1
            while ip2 < parent2.distributionSize() and child.containsProject(parent2.getProject(ip2)):
                ip2 += 1
            x = bool(random.getrandbits(1))
            if (x and ip1 < parent1.distributionSize()) or ip2 >= parent2.distributionSize():
                child.setProject(i, parent1.getProject(ip1))
                ip1 += 1
            else:
                child.setProject(i, parent2.getProject(ip2))
                ip2 += 1
        return child

    def crossoverV3(self, parent1, parent2):
        child = ProjectsManager.Distribution(self.distributionManager)

        n = random.randrange(int(parent1.distributionSize() / 3), int(parent1.distributionSize() / 3 * 2))
        arr = []
        while len(arr) < n:
            x = random.randrange(0, parent1.distributionSize())
            if x not in arr:
                arr.append(x)
        for i in arr:
            child.setProject(i, parent1.getProject(i))
        for i in range(0, parent2.distributionSize()):
            if not child.containsProject(parent2.getProject(i)):
                for ii in range(0, child.distributionSize()):
                    if child.getProject(ii) is None:
                        child.setProject(ii, parent2.getProject(i))
                        break
        return child
