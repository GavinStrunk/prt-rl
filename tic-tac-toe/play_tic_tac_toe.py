'''
Created on Nov 14, 2017

@author: gstrunk
'''
from __future__ import print_function, division

from tic_tac_toe import Environment

class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha

    def play_action(self, env):
        pass
    
    def update_state_history(self,state):
        pass
    
    def value_function_update(self, env):
        pass
    
class Human:
    def __init__(self):
        pass
    
    def set_piece(self, piece):
        self.piece = piece
    
    def play_action(self, env):
        while True:
            move = input("Enter the location of next move row,column (0..2) ex 1,1: ")
            x, y = move.split(',')
            x = int(x)
            y = int(y)
            
            #place piece on board
            break

def play_tic_tac_toe(p1, p2, env, draw=False):
    
    current_player = None
    
    while not env.game_over():
        if current_player == p1:
            current_player = p2
        else:
            current_player = p1
        
        if draw:
            env.draw_board()
        
        current_player.play_action(env)
        
        state = env.get_state()
        p1.update_state_history(state)
        p2.update_state_history(state)
        
    
    p1.value_function_update(env)
    p2.value_function_update(env)
        

if __name__ == '__main__':
    agent1 = Agent()
    agent2 = Agent()
    
    env = Environment()
    
    human = Human()
    