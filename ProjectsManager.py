import random,numpy as np


class Project:
    # TODO To increase efficiency of class
    def __init__(self, pid):
        self.id = pid

    def getID(self):
        return self.id

    def __repr__(self):
        return 'id=' + str(self.id)


class DistributionManager:
    projects = []

    def addProject(self, project):
        self.projects.append(project)

    def getProject(self, index):
        return self.projects[index]

    def numberOfProjects(self):
        return len(self.projects)


class Distribution:
    def __init__(self, distributionManager, distribution=None):
        self.distributionManager = distributionManager
        self.distribution = []
        self.fitness = 0.0
        if distribution is not None:
            self.distribution = distribution
        else:
            for i in range(0, self.distributionManager.numberOfProjects()):
                self.distribution.append(None)

    def __len__(self):
        return len(self.distribution)

    def __getitem__(self, index):
        return self.distribution[index]

    def __setitem__(self, key, value):
        self.distribution[key] = value

    def __repr__(self):
        geneString = "|"
        for i in range(0, self.distributionSize()):
            geneString += str(self.getProject(i)) + "|"
        return geneString

    def generateIndividual(self):
        for projectIndex in range(0, self.distributionManager.numberOfProjects()):
            self.setProject(projectIndex, self.distributionManager.getProject(projectIndex))
        random.shuffle(self.distribution)

    def getProject(self, projectIndex):
        return self.distribution[projectIndex]

    def setProject(self, projectIndex, project):
        self.distribution[projectIndex] = project
        self.fitness = -1

    def getFitness(self,students):
        if self.fitness == -1:
            fitnesses=[]
            for i in range(len(students)):
                fitnesses.append(students[i].calculateCompatibility(self.getProject(i)))
            self.fitness = np.sum(fitnesses)*len(students)
            avg=self.fitness/len(students)
            for x in fitnesses:
                self.fitness -= abs(avg-x)
            self.fitness-=(len(fitnesses)-np.count_nonzero(fitnesses))*10
        return self.fitness

    def print(self,students):
        fitnesses = []
        for i in range(len(students)):
            fitnesses.append(students[i].calculateCompatibility(self.getProject(i)))
        print("First:", len(list(filter(lambda x: x == 9, fitnesses))))
        print("Second:", len(list(filter(lambda x: x == 4, fitnesses))))
        print("Third:", len(list(filter(lambda x: x == 1, fitnesses))))
        print("Zeros:", len(list(filter(lambda x: x == -10, fitnesses))))

    def distributionSize(self):
        return len(self.distribution)

    def containsProject(self, project):
        return project in self.distribution