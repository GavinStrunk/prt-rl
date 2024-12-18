import random
from typing import Tuple, Optional
from prt_sim.jhu.base import BaseEnvironment

class CardDeck:
    """For shuffling and dealing cards"""

    def __init__(self):

        self.cards = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5,
                      6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 10,
                      10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        self.deal_seq = []

    def shuffle_cards(self):
        self.deal_seq = random.sample(self.cards, 24)

    def deal_card(self):
        return self.deal_seq.pop(0)


class Blackjack(BaseEnvironment):
    r"""
    Blackjack simulation class

    """
    def __init__(self):

        self.deck = CardDeck()
        self.agent_total = 0
        self.usable_ace = 0
        self.dealer_card = 0
        self.dealer_total = 0
        self.dealer_ace = 0
        self.current_state = 0
        self.num_states = 203
        self.num_actions = 2

    def get_number_of_states(self) -> int:
        return self.num_states

    def get_number_of_actions(self) -> int:
        return self.num_actions



    def reset(self,
              seed: Optional[int] = None,
              randomize_start: Optional[bool] = False
              ) -> int:
        assert not randomize_start, "Randomizing the start is not supported"
        if seed is not None:
            random.seed(seed)

        self.deck.shuffle_cards()
        self.agent_total = 0
        self.usable_ace = 0
        self.dealer_card = 0
        self.dealer_total = 0
        self.dealer_ace = 0
        self.current_state = 0

        # deal a face up card and a second card to the dealer
        self.dealer_card = self.deck.deal_card()
        d_card_2 = self.deck.deal_card()
        self.dealer_total = self.dealer_card + d_card_2
        if self.dealer_card == 1 or d_card_2 == 1:
            self.dealer_ace = 1
            self.dealer_total += 10
        # print("Dealer has", self.dealer_card, "and", d_card_2)
        # print("Dealer has", self.dealer_total, "points.")

        # deal two cards to the agent
        card_1 = self.deck.deal_card()
        card_2 = self.deck.deal_card()
        self.agent_total = card_1 + card_2
        if card_1 == 1 or card_2 == 1:
            self.usable_ace = 1
            self.agent_total += 10
        # print("Agent has", card_1, "and", card_2)
        # print("Agent has", self.agent_total, "points.")

        # check to see if the agent has a natural (ace + face card)
        if self.agent_total == 21:
            if self.dealer_total == 21:
                self.current_state = 202    # tie game
            else:
                self.current_state = 203    # agent wins

        # otherwise, deal enough cards to the agent so that the total is >11
        else:
            while self.agent_total < 12:
                new_card = self.deck.deal_card()
                self.agent_total += new_card
                if new_card == 1 and self.usable_ace == 0 and self.agent_total < 12:
                    self.usable_ace = 1
                    self.agent_total += 10
                print("Agent drew a", new_card, "and now has", self.agent_total, "points.")
            # now determine the initial state
            self.current_state = self.get_state_index()

        # reset complete; return the initial state
        return self.current_state

    # Use the agent's action to determine the next state and reward
    def execute_action(self,
                       action: int
                       ) -> Tuple[int, float, bool]:
        # action is 'stick'
        if action == 0:
            # dealer's turn
            while self.dealer_total < 17:
                new_card = self.deck.deal_card()
                self.dealer_total += new_card
                if new_card == 1 and self.dealer_ace == 0 and self.dealer_total < 12:
                    self.dealer_ace = 1
                    self.agent_total += 10
                if self.dealer_total > 21 and self.dealer_ace == 1:
                    self.dealer_ace = 0
                    self.agent_total -= 10
            if self.dealer_total > 21:
                # dealer busted; agent wins
                new_state = 203
                reward = 1
                game_end = True
            else:
                if self.dealer_total > self.agent_total:
                    # dealer wins
                    new_state = 201
                    reward = -1
                    game_end = True
                elif self.dealer_total < self.agent_total:
                    # agent wins
                    new_state = 203
                    reward = 1
                    game_end = True
                else:
                    # tie
                    new_state = 202
                    reward = 0
                    game_end = True

        # action is 'hit'
        elif action == 1:
            new_state = self.get_next_state()
            if new_state == 201:
                reward = -1
                game_end = True
            else:
                reward = 0
                game_end = False

        # print("new_state =", new_state, "reward = ", reward, "game_end =", game_end)
        self.current_state = new_state
        return new_state, float(reward), game_end

    def get_state(self):
        return self.current_state

    def get_state_index(self):
        a_idx = self.agent_total - 12
        d_idx = 10 * (self.dealer_card - 1)
        u_idx = 100 * self.usable_ace
        return a_idx + d_idx + u_idx

    def get_next_state(self):
        new_card = self.deck.deal_card()
        self.agent_total += new_card
        if self.agent_total > 21 and self.usable_ace == 1:
            self.usable_ace = 0
            self.agent_total -= 10
        # print("Agent drew a", new_card, "and now has", self.agent_total, "points.")
        if self.agent_total > 21:
            new_state = 201      # 201 is the losing state
        else:
            new_state = self.get_state_index()
        return new_state


