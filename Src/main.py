import tkinter as tk
from tkinter import simpledialog

ROOT = tk.Tk()

ROOT.withdraw()
# the input dialog
USER_INP = simpledialog.askstring(title="Be a playwriter!",
                                  prompt="Remind me how the play starts :")

# check it out
print("Hello", USER_INP)