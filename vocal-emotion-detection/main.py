import os
import keras
import librosa
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


def generate_spectrogram(wavefile_path):
    sampling_rate, x = scipy.io.wavfile.read(wavefile_path)
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


def generate_label_wavefiles(wavfile_path):
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


def generate_dir_labels(wavfile_dir):
    labels = []
    for filename in os.listdir(wavfile_dir):
        label = generate_label_wavefiles(wavfile_dir + "/" + filename)
        labels.append(label)
    return labels


# mylist = os.listdir("dataset/Actor_01")
# print(mylist)


# model = generate_model()

# DIR_PATH = "dataset/Actor_01"

# # label = generate_dir_labels(DIR_PATH)
# # pprint(label)

from pprint import pprint


def generate_feature(wavfile_path):
    x, sampling_rate = librosa.load(
        wavfile_path, res_type="kaiser_fast", duration=2.5, sr=22050 * 2, offset=0.5
    )
    mfccs = librosa.feature.mfcc(y=x, sr=sampling_rate, n_mfcc=13).T
    feature = np.mean(mfccs, axis=0)
    return feature


WAVE_PATH = "dataset/Actor_01/03-01-01-01-01-01-01.wav"

feature = generate_feature(WAVE_PATH)
print(feature)


# sampling_rate, x = scipy.io.wavfile.read(WAVE_PATH)
# image = generate_spectrogram(sampling_rate=sampling_rate, x=x)

# plt.imshow(image, interpolation="nearest", origin="lower", aspect="auto")

# plt.show()
