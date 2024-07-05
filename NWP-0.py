"""
Next Word Generation
Libraries are imported
"""

import tkinter as tk
from tkinter import filedialog, Text

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
import pickle
import numpy as np
import os


"""Load and Pre-process the data"""

file = open("text file.txt", "r", encoding = "utf8")

# store file in list
lines = []
for i in file:
    lines.append(i)

# Convert list to string
data = ""
for i in lines:
  data = ' '. join(lines) 

#replace unnecessary stuff with space
data = data.replace('\n', '').replace('\r', '').replace('\ufeff', '').replace('“','').replace('”','')  
#new line, carriage return, unicode character --> replace by space

#remove unnecessary spaces 
data = data.split()
data = ' '.join(data)
data[:500]

len(data)

"""Tokenizing"""

tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# saving the tokenizer for predict function
pickle.dump(tokenizer, open('token.pkl', 'wb'))

sequence_data = tokenizer.texts_to_sequences([data])[0]
sequence_data[:15]

len(sequence_data)

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)   #unique words

sequences = []

for i in range(3, len(sequence_data)):
    words = sequence_data[i-3:i+1] #Taking 3 words to predict 4th word
    sequences.append(words)
    
print("The Length of sequences are: ", len(sequences))
sequences = np.array(sequences)
sequences[:10]

X = []
y = []
#Seperating the input and output
for i in sequences:
    X.append(i[0:3])
    y.append(i[3])
    
X = np.array(X)
y = np.array(y)

print("Data: ", X[:10])
print("Response: ", y[:10])

y = to_categorical(y, num_classes=vocab_size)
y[:5]

"""Creating model"""

model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=3))
model.add(LSTM(1000, return_sequences=True))
model.add(LSTM(1000))
model.add(Dense(1000, activation="relu"))
model.add(Dense(vocab_size, activation="softmax"))

model.summary()

'''

from tensorflow import keras
from keras.utils.vis_utils import plot_model

keras.utils.plot_model(model, to_file='plot.png', show_layer_names=True)
from tensorflow.keras.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint("next_words.h5", monitor='loss', verbose=1, save_best_only=True)
model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
model.fit(X, y, epochs=10, batch_size=14, callbacks=[checkpoint])
'''

from tensorflow.keras.models import load_model
import numpy as np
import pickle

# Load the model and tokenizer
model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

def Predict_Next_Words(model, tokenizer, word):

  sequence = tokenizer.texts_to_sequences([word])
  sequence = np.array(sequence)
  preds = np.argmax(model.predict(sequence))
  predicted_word = ""

  for key, value in tokenizer.word_index.items():
      if value == preds:
          predicted_word = key
          break

  print(predicted_word)
  return predicted_word


root = tk.Tk()

canvas = tk.Canvas(root, background="#46C7C7" , height=400 , width=700 )
canvas.pack()

frame = tk.Frame(root, bg = "white")
frame.place(relwidth=0.7 , relheight=0.7 , relx=0.1 , rely=0.1)

def printOutput():

        inp = inputtxt.get(1.0, "end-1c")
        lbl.config(text = "provided Input: "+inp)

        l = tk.Label(text ="The next word")
        l.pack()
        word = inp

        if word == "0":
            Output.insert( 'Execution completed........')
            Output.pack()
            #print("Execution completed.....")

        else:
            try:
                word = word.split(" ")
                word = word[-3:]
                print(word)

                Predict_Next_Words(model, tokenizer, word)

            except Exception as e:
                Output.insert(e,'Execution completed........')
                Output.pack()
                #print("Error occurred: ", e)


# Text box creation
inputtxt = tk.Text(frame,height =5,width=20)
inputtxt.pack()

#Button and Label Creation
printButton = tk.Button(frame, text = "print", padx=10 , pady=5 , fg="white" , bg="#2B60DE", command= printOutput)
printButton.pack()
Display = tk.Button(frame, text = "Display", padx=10 , pady=5 , fg="white" , bg="#2B60DE", command= lambda : printOutput())
Display.pack()
Output = Text(frame,height =5,width=10)
Output.pack()
lbl = tk.Label(root, text = "Predict the word")
lbl.pack()
Display.pack()
frame.mainloop()

root.mainloop()
