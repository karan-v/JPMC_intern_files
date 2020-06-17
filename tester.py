from subprocess import call
import os
from extractor import Extractor
from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from data import DataSet
import numpy as np
from natsort import realsorted, ns
import natsort

#Importing CNN Model
extractor_model = Extractor()
#Importing LSTM Model
model = load_model('data/checkpoints/lstm-features.127-0.090.hdf5')
'''
src = "prediction.avi"
dest = os.path.join("tests", "frames",src + '-%04d.jpg')
call(["ffmpeg", "-i", src, dest])
#Extracting CNN features and saving into numpy array
sequence = []
i = 0
sequence_path = "/home/karan/Documents/git_models/five-video-classification-methods/tests/sequence/data_final"
os.chdir("/home/karan/Documents/git_models/five-video-classification-methods/tests/frames/")
sequence = []
files = [f for f in os.listdir('.') if os.path.isfile(f)]
sorted_files = natsort.natsorted(files,reverse=False)
for f in sorted_files:
    i = i + 1
    features = extractor_model.extract(f)
    sequence.append(features)
    if i==40:
        break

np.save(sequence_path, sequence)

'''
sequences = np.load("/home/karan/Documents/git_models/five-video-classification-methods/tests/sequence/data_final.npy")

#Predicting
os.chdir("/home/karan/Documents/git_models/five-video-classification-methods/")
prediction = model.predict(np.expand_dims(sequences, axis=0))

#Importing Classnames
os.chdir("/home/karan/Documents/git_models/five-video-classification-methods/")
from data import DataSet
data = DataSet(seq_length=40, class_limit=4)
def get_classes(self):
    classes = []
    for item in self.data:
        if item[1] not in classes:
            classes.append(item[1])
    # Sort them.
    classes = sorted(classes)
    # Return.
    if self.class_limit is not None:
        return classes[:self.class_limit]
    else:
        return classes

#printing classnames
data.print_class_from_prediction(np.squeeze(prediction, axis=0)) 
