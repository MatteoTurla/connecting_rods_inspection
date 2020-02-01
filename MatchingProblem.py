from search import Problem
import numpy as np
import math

# compute the path between 2 pints using aima library
# we must not open the holes in the blob, so a point on the path is a point that has all the 8 neighborhood different from 0.
class MatchingProblem(Problem):

    def __init__(self, initial, goal, image):
        self.initial = initial
        self.goal = goal
        self.image = image
        Problem.__init__(self, initial, goal)

    def actions(self, state):
        actions = []
        x,y = state
        if self.check_neigh((x+1,y)):
            actions.append((x+1,y))
        if self.check_neigh((x - 1, y)):
            actions.append((x - 1, y))
        if self.check_neigh((x,y+1)):
            actions.append((x,y+1))
        if self.check_neigh((x,y-1)):
            actions.append((x,y-1))
        return actions

    def result(self, state, action):
        return action

    def goal_test(self, state):
        if state == self.goal:
            return True
        else:
            return False

    def check_neigh(self, p):
        # check that after line cut we do not open any hole and we respect the original structure on the blob
        x, y = p
        if np.min(self.image[x - 1:x + 2, y - 1:y + 2]) == 0:
            return False
        return True

    def h(self, node):
        return self.distance(node.state, self.goal)

    def distance(self, p1, p2):
        return self.euclidean_distance(p1,p2)

    def manatthan_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return abs(x1 - x2) + abs(y1 - y2)

    def euclidean_distance(self, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        return math.sqrt((x1-x2)**2+(y1-y2)**2)


