from __future__ import print_function, division

from tic_tac_toe import Environment

class Agent:
    def __init__(self, eps=0.1, alpha=0.5):
        self.eps = eps
        self.alpha = alpha
        self.verbose = False
        self.state_history = []
        
    def set_value(self, value):
        self.value = value
        
    def set_symbol(self, sym):
        self.symbol = sym
        
    def set_verbose(self, verb):
        self.verbose = verb
        
    def reset_history(self):
        self.state_history = []

    def play_action(self, env):
        pass
    
    def update_state_history(self,state):
        self.state_history.append(state)
    
    def update_value_function(self, env):
        #This is the equation used to update the value function
        #V(prev_state) = V(prev_state) + alpha*(V(next_state) - V(prev_state))
        reward = env.reward(self.symbol)
        target = reward
        
        for prev in reversed(self.state_history):
            value = self.value[prev] + self.alpha * (target - self.value[prev])
            self.value[prev] = value
            target = value
        self.reset_history()
    
class Human:
    def __init__(self):
        pass
    
    def set_symbol(self, sym):
        self.symbol = sym
    
    def play_action(self, env):
        while True:
            move = input("Enter the location of next move row,column (0..2) ex 1,1: ")
            x, y = move.split(',')
            x = int(x)
            y = int(y)
            
            env.board.place_piece(self.symbol, x, y)
            break
        
    def update_state_history(self, state):
        pass
    
    def update_value_function(self, env):
        pass

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
    