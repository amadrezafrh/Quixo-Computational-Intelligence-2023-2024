#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 16:49:01 2024

@author: ahmadrezafrh
"""

import tkinter as tk
from tkinter import ttk, messagebox
from ttkbootstrap import Style
from copy import deepcopy
import itertools
from game import Move

class ButMove:
    def __init__(self):
        self.mov = None
        self.pos = None
        self.pos_selected = False
        self.mov_selected = False
        self.pos_ranges = [range(0,5),range(0,5)]
        self.all_pos = list(itertools.product(*self.pos_ranges))
        self.available_pos = []
        self.current_player = 'X'
        self.game = [['', '', '', '', ''] for _ in range(5)]
        for pos in self.all_pos:
            row, col = pos
            from_border = row in (0, 4) or col in (0, 4)
            if from_border:
                self.available_pos.append(pos)
                
                
    def make_pos(self, i, j):
        if self.pos==(i,j):
            self.game[i][j] = ''
            buttons[i][j].configure(text='')
            self.pos = None
            self.pos_selected = False
            if not self.mov_selected:
                self.deeliminate_mov((i,j))
            
        elif self.pos==None and not self.pos_selected:
            if self.game[i][j] == '' or self.game[i][j] == self.current_player:
                if i in (0,4) or j in (0,4):
                    self.game[i][j] = self.current_player
                    buttons[i][j].configure(text=self.current_player, )
                    self.pos = (i, j)
                    self.pos_selected=True
                    if not self.mov_selected:
                        self.eliminate_mov(self.pos)
                    if self.mov:
                        self.act()
                    
    def deeliminate_mov(self, pos):
        axis_0 = pos[0]
        axis_1 = pos[1]
        
        if axis_0 == 0:  # can't move upwards if in the top row...
            buttons[5][0].configure(text=moves[0], )
        elif axis_0 == 4:
            buttons[6][0].configure(text=moves[1], )

        if axis_1 == 0:
            buttons[7][0].configure(text=moves[2], )
        elif axis_1 == 4:
            buttons[8][0].configure(text=moves[3], ) 
            
            
    def eliminate_mov(self, pos):
        axis_0 = pos[0]
        axis_1 = pos[1]
        
        if axis_0 == 0:  # can't move upwards if in the top row...
            buttons[5][0].configure(text="-NOT-SELECTABLE-", )
        elif axis_0 == 4:
            buttons[6][0].configure(text="-NOT-SELECTABLE-", )

        if axis_1 == 0:
            buttons[7][0].configure(text="-NOT-SELECTABLE-", )
        elif axis_1 == 4:
            buttons[8][0].configure(text="-NOT-SELECTABLE-", )
    
    def eliminate_pos(self, mov):
        self.prev_values = deepcopy(self.game)
        for pos in self.available_pos:
            axis_0 = pos[1]
            axis_1 = pos[0]
            if mov == 0 and axis_0 == 0:
                buttons[axis_0][axis_1].configure(text="--", )
            elif mov == 1 and axis_0 == 4:
                buttons[axis_0][axis_1].configure(text="--", )

            if mov == 2 and axis_1 == 0:
                buttons[axis_0][axis_1].configure(text="--", )
            elif mov == 3 and axis_1 == 4:
                buttons[axis_0][axis_1].configure(text="--", )
        
    def deeliminate_pos(self, mov):
        for pos in self.available_pos:
            axis_0 = pos[1]
            axis_1 = pos[0]
            if mov == 0 and axis_0 == 0:
                self.game[axis_0][axis_1] = self.prev_values[axis_0][axis_1]
                buttons[axis_0][axis_1].configure(text=self.prev_values[axis_0][axis_1])
            elif mov == 1 and axis_0 == 4:
                self.game[axis_0][axis_1] = self.prev_values[axis_0][axis_1]
                buttons[axis_0][axis_1].configure(text=self.prev_values[axis_0][axis_1])

            if mov == 2 and axis_1 == 0:
                self.game[axis_0][axis_1] = self.prev_values[axis_0][axis_1]
                buttons[axis_0][axis_1].configure(text=self.prev_values[axis_0][axis_1])
            elif mov == 3 and axis_1 == 4:
                self.game[axis_0][axis_1] = self.prev_values[axis_0][axis_1]
                buttons[axis_0][axis_1].configure(text=self.prev_values[axis_0][axis_1])
                
    def make_mov(self, i,j):
        
        if self.mov==(i,j):
            buttons[5+i][0].configure(text=f"{moves[i]}")
            self.mov = None
            self.mov_selected = False
            if not self.pos_selected:
                self.deeliminate_pos(i)
                
        elif self.mov==None and self.mov!=(i,j) and not self.mov_selected:
            self.mov = (i, j)
            buttons[5+i][0].configure(text=f"{moves[i]}\n-SELECTED-", )
            self.mov_selected = True
            if not self.pos_selected:
                self.eliminate_pos(self.mov[0])
            if self.pos:
                self.act()
            
        
    def act(self):
        self.deeliminate_mov(self.pos)
        self.deeliminate_pos(self.mov[0])
        row = self.pos[0]
        col = self.pos[1]
        self.game[row][col] = self.current_player
        
        
        buttons[row][col].configure(text=self.current_player)
        self.check_winner()
        self.current_player = "O" if self.current_player == "X" else "X"
        self.mov = None
        self.pos = None
        self.pos_selected = False
        self.mov_selected = False
        
        

    def check_winner(self):
        # Check rows, columns and diagonals
        winning_combinations = (self.game[0], self.game[1], self.game[2],self.game[3], self.game[4],
                                [self.game[i][0] for i in range(5)],
                                [self.game[i][1] for i in range(5)],
                                [self.game[i][2] for i in range(5)],
                                [self.game[i][3] for i in range(5)],
                                [self.game[i][4] for i in range(5)],
                                [self.game[i][i] for i in range(5)],
                                [self.game[i][2 - i] for i in range(5)])

        for combination in winning_combinations:
            if combination[0] == combination[1] == combination[2] == combination[3] == combination[4] != '':
                self.announce_winner(combination[0])

        if all(self.game[i][j] != '' for i in range(5) for j in range(5)):
            self.announce_winner("Draw")

    def announce_winner(self, player):
        if player == "Draw":
            message = "It's a draw!"
        else:
            message = f"Player {player} wins!"
        messagebox.showinfo("Game Over", message)
        self.reset_game()

    def reset_game(self):
        self.game = [['', '', '', '', ''] for _ in range(5)]
        self.current_player = "X"
        for row in buttons:
            for button in row:
                button.configure(text='')




if __name__=="__main__":

    root = tk.Tk()
    root.title("Quixo")
    style = Style(theme="flatly")
    
    buttons = []
    mov_obj = ButMove()
    for i in range(5):
        row = []
        for j in range(5):
            button = tk.Button(root, text='', width=5,
                                command=lambda i=i, j=j: mov_obj.make_pos(i,j))
            button.grid(row=i, column=j, padx=5, pady=5)
            row.append(button)
        buttons.append(row)
    
    
    
    moves = ['TOP', 'BOTTOM', 'LEFT', 'RIGHT']
    actions = {
        moves[0] : Move.TOP,
        moves[1] : Move.BOTTOM,
        moves[2] : Move.LEFT,
        moves[3] : Move.RIGHT,
        
    }
    
    for i in range(4):
        row = []
        button = tk.Button(root, text=f'{moves[i]}', width=20,
                            command=lambda i=i, j=7: mov_obj.make_mov(i,j), bg='RED', fg='yellow')
        button.grid(row=i, column=7, padx=40, pady=7)
        row.append(button)
        buttons.append(row)
    
    mov_obj.current_player = "X"
    root.mainloop()