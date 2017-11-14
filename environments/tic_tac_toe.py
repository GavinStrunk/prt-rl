from __future__ import print_function

import numpy as np
from PyQt5.QtWidget import QWidget

class Environment:
	def __init__(self):
		self.state = None
	
	def game_over(self):
		return True
	
	def draw_board(self):
		pass
	
	def get_state(self):
		return self.state
	
class Board:
	#Values for pieces
	PIECE_X = 2
	PIECE_O = 3

	def __init__(self):
		self.state = np.zeros((3,3), dtype=int)
		self.place_cnt = 0
		self.row_sum = np.zeros(3)
		self.column_sum = np.zeros(3)
		self.pos_diag_sum = 0
		self.neg_diag_sum = 0

	def place_piece(self, piece, x, y):
		#piece is either X or O
		#x is the row location and y is the z
		if (piece != self.PIECE_X and piece != self.PIECE_O):
			print("Invalid piece selection")
			return -1

		if (x < 0 or x > 2):
			print("Out of bounds row selection")
			return -1

		if (y < 0 or y > 2):
			print("Out of bounds column selection")
			return -1

		if self.state[x,y] != 0:
			print("Already a piece in that location")
			return -1

		self.state[x,y] = piece

		#update variables used to check if game is over
		self.place_cnt += 1
		self.row_sum[x] += piece
		self.column_sum[y] += piece

		if x == y:
			self.neg_diag_sum += piece

		if x + y == 2:
			self.pos_diag_sum += piece

		return 0

	def check_for_game_over(self):
		#check rows
		for i in np.nditer(self.row_sum):
			if i == 6:
				print("X's Win")
				return 2
			elif i == 9:
				print("O's Win")
				return 3

		#check columns
		for i in np.nditer(self.column_sum):
			if i == 6:
				print("X's Win")
				return 2
			elif i == 9:
				print("O's Win")
				return 3

		#check diags
		if self.pos_diag_sum == 6:
			print("X's Win")
			return 2
		elif self.pos_diag_sum == 9:
			print("O's Win")
			return 3

		if self.neg_diag_sum == 6:
			print("X's Win")
			return 2
		elif self.neg_diag_sum == 9:
			print("O's Win")
			return 3

		#Check for draw
		if self.place_cnt == 9:
			print("Draw")
			return 1
		
		return 0

	def print_board(self):
		#print with underline so it looks like a board
		print('\033[4m%s|%s|%s' % (self._state_to_str(self.state[0,0]),
							self._state_to_str(self.state[0,1]),
							self._state_to_str(self.state[0,2])))
		print('\033[4m%s|%s|%s' % (self._state_to_str(self.state[1,0]),
							self._state_to_str(self.state[1,1]),
							self._state_to_str(self.state[1,2])))

		#end of the underlining
		print('\033[0m%s|%s|%s' % (self._state_to_str(self.state[2,0]),
							self._state_to_str(self.state[2,1]),
							self._state_to_str(self.state[2,2])))

	def _state_to_str(self, state):
		if state == self.PIECE_X:
			sstr = 'X'
		elif state == self.PIECE_O:
			sstr = 'O'
		else:
			sstr = ' '

		return sstr


def board_tests():
	board = Board()
	board.print_board()

	#place x
	board.place_piece(board.PIECE_X, 1, 1)
	board.print_board()

	board.place_piece(board.PIECE_O, 2, 2)
	board.print_board()

	#check boundary conditions
	#All of these should print errors
	board.place_piece(0, 0, 0)
	board.place_piece(board.PIECE_X, -1, 0)
	board.place_piece(board.PIECE_X, 3, 0)
	board.place_piece(board.PIECE_X, 1, -1)
	board.place_piece(board.PIECE_X, 1, 3)
	board.place_piece(board.PIECE_O, 1, 1)

	#check an x row win
	board.place_piece(board.PIECE_X, 0, 1)
	board.place_piece(board.PIECE_X, 2, 1)
	board.print_board()
	board.check_for_game_over()

	board2 = Board()
	board2.place_piece(board.PIECE_O, 0, 2)
	board2.place_piece(board.PIECE_O, 1, 2)
	board2.place_piece(board.PIECE_O, 2, 2)
	board2.print_board()
	board2.check_for_game_over()

	board3 = Board()
	board3.place_piece(board.PIECE_O, 0, 0)
	board3.place_piece(board.PIECE_O, 1, 1)
	board3.place_piece(board.PIECE_O, 2, 2)
	board3.print_board()
	board3.check_for_game_over()	

	board4 = Board()
	board4.place_piece(board.PIECE_O, 0, 0)
	board4.place_piece(board.PIECE_O, 1, 1)
	board4.place_piece(board.PIECE_O, 2, 2)
	board4.print_board()
	board4.check_for_game_over()

	board4 = Board()
	board4.place_piece(board.PIECE_O, 2, 0)
	board4.place_piece(board.PIECE_O, 0, 2)
	board4.place_piece(board.PIECE_O, 1, 1)
	board4.print_board()
	board4.check_for_game_over()

if __name__ == '__main__':

	board_tests()
	