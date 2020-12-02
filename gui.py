import os

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import tkinter as tk
from PIL import ImageGrab
import math
import numpy as np
import matplotlib.pyplot as plt
import keras
#import tensorflow as tf
#from tensorflow import keras

model = keras.models.load_model('./models')

def handle_reset():
    print("reset")
    canvas.delete("all")
    global image
    image = np.zeros((28, 28), dtype=np.uint8)

def handle_what():
    global image
    test = image.reshape((1, 28 * 28))
    print(image)
    test = test.astype('float32') / 255
    predictions = model.predict(test)
    print(predictions)
    print(np.argmax(predictions))
    

def position(event):
    #print(event.x, event.y)
    x, y = math.floor(event.x/10), math.floor(event.y/10)
    canvas.create_rectangle(10*x, 10*y, 10*x+10, 10*y+10, fill="black")
    global lastx, lasty, image
    lastx, lasty = x, y
    image[x, y] = 255

def draw(event):
    #print(event.x, event.y)
    global lastx, lasty, image
    x, y = math.floor(event.x/10), math.floor(event.y/10)
    canvas.create_rectangle(10*x, 10*y, 10*x+10, 10*y+10, fill="black")
    image[x, y] = 255
    #canvas.create_line(lastx, lasty, event.x, event.y)
    #lastx, lasty = event.x, event.y

window = tk.Tk()

lastx, lasty = 0, 0
image = np.zeros((28,28), dtype=np.uint8)

frame_top = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
frame_top.grid(row=0, column=0)

frame_bot = tk.Frame(
            master=window,
            relief=tk.RAISED,
            borderwidth=1
        )
frame_bot.grid(row=1, column=0)

label = tk.Label(master=frame_top, text="Tian draws", background="purple")
label.pack(side=tk.LEFT)

reset = tk.Button(master=frame_top, text="Reset", command=handle_reset)
reset.pack(side=tk.LEFT)

what = tk.Button(master=frame_top, text="What is it", command=handle_what)
what.pack(side=tk.LEFT)

canvas = tk.Canvas(master=frame_bot, bg="white", height=280, width=280, highlightthickness=0)

# for i in range(28):
#     canvas.create_line(10*i, 0, 10*i, 280)
#     canvas.create_line(0, 10*i, 280, 10*i)

canvas.bind("<Button-1>", position)
canvas.bind("<B1-Motion>", draw)
canvas.pack(fill='both', expand=True)

window.mainloop()