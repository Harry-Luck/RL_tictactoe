from tkinter import *
from main import *
import const

class tile(Button):
    def __init__(self,id,mainframe):
        self.name = str(id)

        super().__init__(mainframe,
                         text = " ",
                         height = 3,
                         width = 7,
                         borderwidth = 0,
                         bg = "Lightgray",
                         font= ("Roboto", 22),
                         command = lambda: self.make_move(id))
    def update(self,player):
        # update the tile according to the player who is playing
        if player == 1:
            self["text"] = "X"
        else:
            self["text"] = "O"

    def make_move(self,action):
        global current_node
        global buttons
        if const.game.is_valid(action) == True:
            self.update(1)
            const.game.invert_board()
            r = - const.game.step(action)
            const.game.invert_board()
            current_node = const.mct[current_node.Child_nodes[action]]
            player = 0
        
            if const.game.terminal == False:
                #player 2 plays
                a = choose_move(current_node)
                buttons[a].update(0)
                r = const.game.step(a)
                current_node = const.mct[current_node.Child_nodes[a]]
                player = 1
        
            if const.game.terminal == True:
                global bottom_label
                for b in buttons:
                    b["state"] = "disabled"
                if r == 0:
                    bottom_label["text"] = "Tie"
                elif r == -1:
                    bottom_label["text"] = "You won"
                else:
                    bottom_label["text"] = "You lost"

def generate_buttons(num_rows, num_cols,mainframe):
    # generate as many buttons as the number of tiles in the board
    buttons = []
    for r in range(num_rows):
        for c in range(num_cols):
            new_button = tile(r*num_cols + c,mainframe)
            buttons.append(new_button)
            new_button.grid(row = r, column = c, padx=(1,1), pady=(1,1))
    return buttons

def restart():
    # restart the game
    global buttons
    global player
    global current_node
    player = random.randint(0,1)    # choose player
    const.game.restart()                  # empty board
    current_node = const.mct[0]   # root of the tree is current node

    if buttons != []:
        for b in buttons:
            b["state"] = "normal"
            b["text"] = " "
        global bottom_label
        bottom_label["text"] = "Make your move!"

# initialize
buttons = []
restart()

# display
root = Tk()
# root.attributes("-fullscreen", True) -- this is not a nice fullscreen
topframe = Frame(root)
topframe.pack()
mainframe = Frame(root)
mainframe.pack()
bottomframe = Frame(root)
bottomframe.pack()

title = Label(topframe,
              text="Tic-Tac-Toe Zero",
              font= ("Roboto", 30))
title.pack(pady = (32,32))
buttons = generate_buttons(3,3,mainframe)
bottom_label = Label(bottomframe,
                     text = "make your move!",
                     font = ("Roboto", 14))
bottom_label.pack(pady = (32,32))
restart_button = Button(bottomframe,
                        text = "Restart",
                        bg = "DarkGray",
                        borderwidth = 0,
                        font = ("Roboto",14),
                        command = restart)
restart_button.pack(pady = (0,32))

#choose move
if player == 0:
    #if player 1 not random
    a = choose_move(current_node)
    buttons[a].update(player)
    r = const.game.step(a)

root.mainloop()



