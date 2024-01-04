import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import Dense
import scipy.io.wavfile
import numpy as np
import matplotlib.pyplot as plt


def generate_model():
    model = Sequential()
    model.add(Conv1D(256, 5, padding="same", input_shape=(216, 1)))
    model.add(Activation("relu"))
    model.add(Conv1D(128, 5, padding="same"))
    model.add(Activation("relu"))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(8)))
    model.add(
        Conv1D(
            128,
            5,
            padding="same",
        )
    )
    model.add(Activation("relu"))
    # model.add(Conv1D(128, 5,padding='same',))
    # model.add(Activation('relu'))
    # model.add(Conv1D(128, 5,padding='same',))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.2))
    model.add(
        Conv1D(
            128,
            5,
            padding="same",
        )
    )
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(10))
    model.add(Activation("softmax"))

    optimizer = keras.optimizers.RMSprop(learning_rate=0.00001, decay=1e-6)
    model.compile(
        loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"]
    )
    model.summary()

    return model


def generate_spectrogram(image_path):
    sampling_rate, x = scipy.io.wavfile.read(image_path)

    nstep = int(sampling_rate * 0.01)
    nwin = int(sampling_rate * 0.03)
    nfft = nwin
    window = np.hamming(nwin)
    nn = range(nwin, len(x), nstep)
    X = np.zeros((len(nn), nfft // 2))

    for i, n in enumerate(nn):
        xseg = x[n - nwin : n]
        z = np.fft.fft(window * xseg, nfft)
        X[i, :] = np.log(np.abs(z[: nfft // 2]))

    return X.T


# model = generate_model()

IMAGE_PATH = "dataset/Actor_01/03-01-01-01-01-01-01.wav"

# image = generate_spectrogram(IMAGE_PATH)

# plt.imshow(image, interpolation="nearest", origin="lower", aspect="auto")

# plt.show()


import os


mylist = os.listdir("dataset/Actor_01")
print(mylist)


def generate_label_waveflie(wavfile_path):
    filename = wavfile_path.split("/")[-1].split(".")[0]
    parameters = filename.split("-")
    emotion_id = int(parameters[2])
    actor_id = int(parameters[6])
    is_female = actor_id % 2 == 0

    emotion = None
    if emotion_id == 1:
        emotion = "neutral"
    elif emotion_id == 2:
        emotion = "calm"
    elif emotion_id == 3:
        emotion = "happy"
    elif emotion_id == 4:
        emotion = "sad"
    elif emotion_id == 5:
        emotion = "angry"
    elif emotion_id == 6:
        emotion = "fearful"
    elif emotion_id == 7:
        emotion = "disgust"
    elif emotion_id == 8:
        emotion = "surprised"

    return {"emotion": emotion, "gender": "female" if is_female else "male"}


label = generate_label_waveflie(IMAGE_PATH)
print(label)
