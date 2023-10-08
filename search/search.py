# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
search.py: generic search algorithms which are called by Pacman agents (in searchAgents.py).
"""

from builtins import object
import util


class SearchProblem(object):
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    Do not change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def generalGraphSearch(problem, structure, greedy=False):
    """
    Defines a general algorithm to search a graph.
    Parameters are structure, which can be any data structure with .push() and .pop() methods, and problem, which is the
    search problem.
    """

    # Push the root node/start into the data structure in this format: [(state, action taken, cost)]
    # The list pushed into the structure for the second node will look something like this:
    # [(root_state, "Stop", 0), (new_state, "North", 1)]
    structure.push([(problem.getStartState(), "Stop", 0)])

    # Initialise the list of visited nodes to an empty list
    visited = []
#    print visited

    # While the structure is not empty, i.e. there are still elements to be searched,

    while not structure.isEmpty():
        # get the path returned by the data structure's .pop() method
        path = structure.pop()

        # The current state is the first element in the last tuple of the path
        # i.e. [(root_state, "Stop", 0), (new_state, "North", 1)][-1][0] = (new_state, "North", 1)[0] = new_state
        curr_state = path[-1][0]

        # if the current state is the goal state,
        if problem.isGoalState(curr_state):
            # return the actions to the goal state
            # which is the second element for each tuple in the path, ignoring the first "Stop"
            return [x[1] for x in path][1:]

        # if the current state has not been visited,
        if curr_state not in visited:
            # mark the current state as visited by appending to the visited list
            visited.append(curr_state)

            # for all the successors of the current state,
            for successor in problem.getSuccessors(curr_state):
                # successor[0] = (state, action, cost)[0] = state
                # if the successor's state is unvisited,
                if successor[0] not in visited:
                    # Copy the parent's path
                    successorPath = path[:]
                    # Set the path of the successor node to the parent's path + the successor node
                    successorPath.append(successor)
                    # Push the successor's path into the structure
                    structure.push(successorPath)

    # If search fails, return False
    return False



def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    The search algorithm returns a list of actions that reaches the
    goal. 

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """

    # Initialize an empty Stack
    stack = util.Stack()

    # DFS is general graph search with a Stack as the data structure
    return generalGraphSearch(problem, stack) 

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""

    # Initialize an empty Queue
    queue = util.Queue()

    # BFS is general graph search with a Queue as the data structure
    return generalGraphSearch(problem, queue)

    
def uniformCostSearch(problem):
    """Search the node of least total cost first. This is Dijkstra"""

    # The cost for UCS only the backward cost
    # get the actions in the path which are the second element for each tuple in the path, ignoring the first "Stop"
    # calculate the cost of the actions specific to the Problem using problem.getCostOfActions
    g = lambda path: problem.getCostOfActions([x[1] for x in path][1:])

    # Construct an empty priority queue that sorts using this backwards cost
    pq = util.PriorityQueueWithFunction((g,))

    # UCS is general graph search with the PriorityQueue sorting by the cost as the data structure
    return generalGraphSearch(problem, pq)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""

    # The cost for a* seach is f(x) = g(x) + h(x)
    # The heuristic is h(x), heuristic(state, problem),
    # where state = path[-1][0], which is the first element in the last tuple of the path
    f = lambda path: problem.getCostOfActions([x[1] for x in path][1:]) + heuristic(path[-1][0], problem)
    h = lambda path: heuristic(path[-1][0], problem)
    
    # Construct an empty priority queue that sorts using f(x) and breaks ties by h(x)
    pq = util.PriorityQueueWithFunction((f,h))

    # A* is general graph search with the PriorityQueue sorting by the f(x) as the data structure
    return generalGraphSearch(problem, pq)




# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch

