# Licensing Information:  You are free to use or extend this codebase for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide the following
# attribution:
# This CSCE-689 RL assignment codebase was developed at Texas A&M University.
# The core code base was developed by Guni Sharon (guni@tamu.edu).

import numpy as np
import heapq
from Solvers.Abstract_Solver import AbstractSolver, Statistics
import os

class ValueIteration(AbstractSolver):

    def __init__(self,env,options):
        assert str(env.observation_space).startswith( 'Discrete' ), str(self) + \
                                                                    " cannot handle non-discrete state spaces"
        assert str(env.action_space).startswith('Discrete'), str(self) + " cannot handle non-discrete action spaces"
        super().__init__(env,options)
        self.V = np.zeros(env.nS)

    def train_episode(self):
        """
            Inputs: (Available/Useful variables)
                self.env
                    this the OpenAI GYM environment
                         see http://gym.openai.com/

                state = self.env.reset():
                    Resets the environment and returns the starting state

                self.env.nS:
                    number of states in the environment

                self.env.nA:
                    number of actions in the environment

                for probability, next_state, reward, done in self.env.P[state][action]:
                    `probability` will be probability of `next_state` actually being the next state
                    `reward` is the short-term/immediate reward for achieving that next state
                    `done` is a boolean of wether or not that next state is the last/terminal state

                    Every action has a chance (at least theortically) of different outcomes (states)
                    Which is why `self.env.P[state][action]` is a list of outcomes and not a single outcome

                self.options.gamma:
                    The discount factor (gamma from the slides)

            Outputs: (what you need to update)
                self.V:
                    This is a numpy array, but you can think of it as a dictionary
                    `self.V[state]` should return a floating point value that
                    represents the value of a state. This value should become
                    more accurate with each episode.

                    How should this be calculated?
                        look at the value iteration algorithm
                        Ref: Sutton book eq. 4.10.
                    Once those values have been updated, thats it for this function/class
        """

        # you can add variables here if it is helpful

        # Update the estimated value of each state
        for each_state in range(self.env.nS):

            ###################################################
            #            Compute self.V here                  #
            # Do a one-step lookahead to find the best action #
            #           YOUR IMPLEMENTATION HERE              #
            ###################################################
            raise NotImplementedError

        # Dont worry about this part
        self.statistics[Statistics.Rewards.value] = np.sum(self.V)
        self.statistics[Statistics.Steps.value] = -1

    def __str__(self):
        return "Value Iteration"

    def create_greedy_policy(self):
        """
        Creates a greedy policy based on state values.
        Use:
            self.env.nA: Number of actions in the environment.
        Returns:
            A function that takes an observation as input and returns a Greedy
               action
        """

        def policy_fn(state):
            """
            What is this function?
                This function is the part that decides what action to take

            Inputs: (Available/Useful variables)
                self.V[state]
                    the estimated long-term value of getting to a state

                self.env.nA:
                    number of actions in the environment
            """

            ################################
            #   YOUR IMPLEMENTATION HERE   #
            ################################
            raise NotImplementedError

        return policy_fn


class PriorityQueue:
    """
    Implements a priority queue data structure. Each inserted item
    has a priority associated with it and the client is usually interested
    in quick retrieval of the lowest-priority item in the queue. This
    data structure allows O(1) access to the lowest-priority item.
    """

    def __init__(self):
        self.heap = []
        self.count = 0

    def push(self, item, priority):
        entry = (priority, self.count, item)
        heapq.heappush(self.heap, entry)
        self.count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.heap)
        return item

    def isEmpty(self):
        return len(self.heap) == 0

    def update(self, item, priority):
        # If item already in priority queue with higher priority, update its priority and rebuild the heap.
        # If item already in priority queue with equal or lower priority, do nothing.
        # If item not in priority queue, do the same thing as self.push.
        for index, (p, c, i) in enumerate(self.heap):
            if i == item:
                if p <= priority:
                    break
                del self.heap[index]
                self.heap.append((priority, c, item))
                heapq.heapify(self.heap)
                break
        else:
            self.push(item, priority)
