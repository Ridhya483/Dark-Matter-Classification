import tensorflow as tf
from tensorflow.keras.models import model_from_json
import pickle

# loading testing data
X = pickle.load(open("X.pickle","rb"))
Y = pickle.load(open("Y.pickle","rb"))

# scaling the data
X = X/255.0

#loading model
json_file = open('./model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# loading weights of model
loaded_model.load_weights("model.h5")
print("\n\nModel loaded from disc")

loaded_model.compile(loss = "binary_crossentropy" , optimizer="adam" , metrics=["accuracy"])

# evaluating our model
score = loaded_model.evaluate(X,Y,verbose=0)
print("\n\nTrain Data Accuracy :")
print("\n\n%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
