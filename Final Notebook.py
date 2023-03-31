

# ### Importing Necessary Modules




from google.colab import drive
drive.mount('/content/drive')





import pandas as pd
import numpy as np
from sklearn import preprocessing

import tensorflow
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Embedding, Flatten, MaxPooling1D, Dropout, Conv1D, Input, LSTM, SpatialDropout1D, Bidirectional
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Embedding

from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import TFDistilBertForSequenceClassification, DistilBertTokenizerFast


# ## Loading the Data File




df = pd.read_csv('/content/drive/MyDrive/FData.csv')
df = df[['text','is_depressed']]
df





df.is_depressed.value_counts()





# preprocess data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data['text'])
X = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(X, maxlen=100)

# define labels
y = pd.get_dummies(data['is_depressed']).values

# split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# ## Model

# The first layer is an Embedding layer with input_dim=5000, output_dim=128, and input_length=100, which means it will convert the input sequence of integers (with values between 0 and 4999) into a dense vector representation of length 128. The input_length is set to 100, which means that the input sequence is expected to have a length of 100 integers.
# 
# The second layer is an LSTM layer with 64 units, which means it has 64 memory cells. The dropout parameter is set to 0.2, which means that 20% of the inputs will be randomly set to 0 during training to prevent overfitting. The recurrent_dropout parameter is also set to 0.2, which means that 20% of the recurrent connections will be randomly set to 0 during training.
# 
# The third and final layer is a Dense layer with 2 units and a sigmoid activation function. This layer will produce two output values between 0 and 1, representing the probabilities of the input sequence belonging to each of the two classes. 
# 
# The model is compiled with binary_crossentropy as the loss function, adam as the optimizer, and accuracy as the metric to be optimized during training.




# build model
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(LSTM(units=64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(units=2, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=104)

# evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print('Test accuracy:', accuracy)


# ### Saving the Model



model.save("Final_model")


# ### Loading the Model




model = keras.models.load_model('Final_model')


# ### Making the Predictions




pexample1 = "As I lay in bed, unable to shake off the heavy weight of despair that seemed to suffocate me, I couldn't help but feel like a mere shadow of the person I used to be, drained of all motivation, hope, and joy."
pexample2 = "I got D grade in my exam even I was fully prepared. I think only solution to get rid of this badluck is to sue myself."

ptest = [pexample1,pexample2]





tokenizer = Tokenizer(num_words=5000)
test=tokenizer.texts_to_sequences(ptest)
test=pad_sequences(test, maxlen=100)
dic={0:'Not-Depressed',1:'Depressed'}

for i in model.predict(test):
    print(dic[np.argmax(i)])







