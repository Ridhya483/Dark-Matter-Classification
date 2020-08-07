import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense , Dropout , Flatten , Conv2D , MaxPooling2D , Activation
import pickle

# LOADING TRAIN DATA
X  = pickle.load(open("X.pickle","rb"))
Y  = pickle.load(open("Y.pickle","rb"))

# SCALING THE FEATURES
X = X/255.0

# INITIALIZING SEQUENTIAL MODEL
model = Sequential()

# ADDING LAYERS
model.add(Conv2D( 64 , (3,3) , input_shape = X.shape[1:] ) )
model.add( Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2) ) )


model.add(Conv2D( 64 , (3,3) ) )
model.add( Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2) ) )

model.add(Conv2D( 64 , (3,3) ) )
model.add( Activation("relu"))
model.add(MaxPooling2D(pool_size = (2,2) ) )

# DROPPING FEATURES FOR PREVENTING OVERFITTING 
model.add(Dropout(0.5))

model.add(Flatten() )

model.add(Dense(64) )
model.add(Activation("tanh"))

model.add( Dense(1) )
model.add(Activation("sigmoid") )

model.compile( loss = "binary_crossentropy", optimizer = "adam" , metrics=["accuracy"])

model.fit(X , Y , batch_size = 50 , epochs= 5 ,validation_split = 0.1)


# SAVING THE MODEL

model_json = model.to_json()
with open("model.json" , "w") as json_file:
    json_file.write(model_json)

model.save_weights("model.h5")
print("\n\nModel has been saved")




                        
