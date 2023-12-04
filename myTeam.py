# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import util

from captureAgents import CaptureAgent
from game import Directions
from util import nearestPoint


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --redOpts and --blueOpts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = float('inf')
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearestPoint(pos):
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        return {'successor_score': 1.0}

class OffensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=.1): #initialize relevant variables
        super().__init__(index, time_for_computing)
        self.state = 'collecting'
        self.defensive_agent = DefensiveReflexAgent(index)
        self.initial_food_amount = None
        self.defender_position = None  

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        food_list = self.get_food(successor).as_list()
        features['successor_score'] = -len(food_list)
        my_state = successor.get_agent_state(self.index)

        if len(food_list) > 0:
            my_pos = successor.get_agent_state(self.index).get_position()
            min_distance = min([self.get_maze_distance(my_pos, food) for food in food_list])
            features['distance_to_food'] = min_distance

        defenders = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        defenders = [a for a in defenders if not a.is_pacman and a.get_position() is not None]

        if len(defenders) > 0:
            self.defender_position = defenders[0].get_position()  #store defender position

        if self.is_dead_end(successor): #avoid dead_ends when escaping from the deffenders
            features['dead_end'] = 1
        else:
            features['dead_end'] = 0

        if self.is_not_losing(game_state) and self.is_time_80_percent_consumed(game_state):  #switch to defend if we not loosing and the game is finishing 
            features['switch_to_defense'] = 1
        else:
            features['switch_to_defense'] = 0

        return features

    
    def choose_action(self, game_state):
        actions = game_state.get_legal_actions(self.index)

        if self.is_defender_close(game_state):     #check if the agent is being chased
            if not hasattr(self, 'evading_defender') or not self.evading_defender:
                self.evading_defender = True
                self.evade_defender_timer = 50  

            if self.evade_defender_timer > 0: #go away from the defender if being chased
                closest_defender_pos = self.get_closest_defender_position(game_state)
                if closest_defender_pos is not None:
                    away_actions = [a for a in actions if
                                    self.get_maze_distance(game_state.get_agent_position(self.index),closest_defender_pos)
                                    <
                                    self.get_maze_distance(self.get_successor(game_state, a).get_agent_position(self.index),closest_defender_pos)]
                    if away_actions:
                        return random.choice(away_actions)

            # If evasion timeout or unable to move away, reset evasion state
            self.evading_defender = False
            self.evade_defender_timer = 0

        #if not being chased, default behavior
        values = [self.evaluate(game_state, a) for a in actions]
        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if self.initial_food_amount is None:
            self.initial_food_amount = food_left

        if food_left < int(self.initial_food_amount * 0.64):   #after eating 36% of food, agent returns to own half to secure the points
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.defender_position, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist

            if successor.get_agent_position(self.index) == self.defender_position: #after securing them, eat the 36% of remaining food
                self.initial_food_amount = food_left

            return best_action

        
        if self.is_defender_close(game_state): #check if defender is close
            if self.state == 'collecting':
                my_pos = game_state.get_agent_position(self.index)
                return self.return_to_base(game_state, actions, my_pos)
            else:
                self.state = 'returning'
                my_pos = game_state.get_agent_position(self.index)
                return self.return_to_base(game_state, actions, my_pos)

        if self.should_switch_to_defense(game_state):
            return self.defensive_agent.choose_action(game_state)
        return random.choice(best_actions)
        
    
    def at_start_position(self, game_state):
        """
        Check if the agent is at the start position.
        """
        my_pos = game_state.get_agent_position(self.index)
        return my_pos == self.start

    def should_switch_to_defense(self, game_state):
        return self.is_not_losing(game_state) and self.is_time_80_percent_consumed(game_state)

    def choose_best_return_action(self, game_state, actions, my_pos):
        closest_defender_pos = self.get_closest_defender_position(game_state)
        if closest_defender_pos is not None:  
            action_distances = [
                (action, self.get_maze_distance(my_pos, self.start), self.get_maze_distance(my_pos, closest_defender_pos))
                for action in actions
            ]
            best_action, _, _ = max(action_distances, key=lambda x: (x[1], -x[2]))
            return best_action
        else:
            return random.choice(actions) 


    def return_to_base(self, game_state, actions, my_pos):
        my_pos = self.start
        return self.choose_best_return_action(game_state, actions, my_pos)

    def is_dead_end(self, game_state):
        my_state = game_state.get_agent_state(self.index)
        my_pos = my_state.get_position()
        legal_actions = game_state.get_legal_actions(self.index)
        legal_actions.remove(Directions.STOP)

        num_legal_actions = len(legal_actions)
        if num_legal_actions == 1:
            next_action = legal_actions[0]
            next_state = self.get_successor(game_state, next_action)
            next_legal_actions = next_state.get_legal_actions(self.index)
            return len(next_legal_actions) == 1

        return False

    def get_weights(self, game_state, action):
        return {'successor_score': 100, 'distance_to_food': -1, 'distance_to_defender': 200000000,
                'dead_end': -5000000000, 'switch_to_defense': 50000000000000000}

    def has_score_increased(self, game_state, amount):
        my_score = game_state.get_score()
        last_score = self.last_score if hasattr(self, 'last_score') else my_score
        if my_score > last_score:
            self.last_score = my_score
            return True
        return False

    def is_defender_close(self, game_state):
        defenders = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in defenders if not a.is_pacman and a.get_position() is not None]
        if len(defenders) > 0:
            my_pos = game_state.get_agent_position(self.index)
            dists_to_defenders = [self.get_maze_distance(my_pos, a.get_position()) for a in defenders]
            return min(dists_to_defenders) < 5
        return False

    def get_closest_defender_position(self, game_state):
        defenders = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        defenders = [a for a in defenders if not a.is_pacman and a.get_position() is not None]
        if len(defenders) > 0:
            closest_defender = min(defenders, key=lambda x: self.get_maze_distance(game_state.get_agent_position(self.index), x.get_position()))
            return closest_defender.get_position()
        return None

    def is_not_losing(self, game_state):
        my_score = game_state.get_score()
        return my_score >= 0

    def is_time_80_percent_consumed(self, game_state):
        configuration = game_state.get_agent_state(self.index).configuration
        if hasattr(configuration, 'timeleft'):
            current_time = configuration.timeleft
            total_time = game_state.get_max_time()
            return current_time < 0.2 * total_time
        else:
            return False

class DefensiveReflexAgent(ReflexCaptureAgent):
    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [a for a in enemies if a.is_pacman and a.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)

            scared_invaders = [a for a in invaders if a.scared_timer > 0]
            if len(scared_invaders) > 0:
                features['invader_scared'] = 1
            else:
                features['invader_scared'] = 0

        if self.is_losing(game_state) and self.is_time_80_percent_consumed(game_state):
            features['switch_to_attack'] = 1
        else:
            features['switch_to_attack'] = 0

        if action == Directions.STOP:
            features['stop'] = 1

        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        return features

    def get_weights(self, game_state, action):
        return {
            'num_invaders': -1000,
            'on_defense': 1000,
            'invader_distance': -10,
            'invader_scared': -20000000000000,
            'stop': -100,
            'reverse': -2,
            'switch_to_attack': 20000
        }

    def is_losing(self, game_state):
        my_score = game_state.get_score()
        if my_score <= 0:
            return True
        else:
            return False

    def is_time_80_percent_consumed(self, game_state):
        configuration = game_state.get_agent_state(self.index).configuration
        if hasattr(configuration, 'timeleft'):
            current_time = configuration.timeleft
            total_time = game_state.get_max_time()
            return current_time < 0.2 * total_time
        else:
            return False