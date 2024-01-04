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
from tqdm import tqdm
import pandas as pd
from sklearn.utils import shuffle


def generate_feature(wavfile_path):
    x, sampling_rate = librosa.load(
        wavfile_path, res_type="kaiser_fast", duration=2.5, sr=22050 * 2, offset=0.5
    )
    sampling_rate = np.array(sampling_rate)
    mfccs = librosa.feature.mfcc(y=x, sr=sampling_rate, n_mfcc=13)
    feature = np.mean(mfccs, axis=0)
    return feature


def generate_dir_features(wavfile_dir):
    data = []
    print("\n[*] Generating features for directory: " + wavfile_dir)
    for filename in tqdm(os.listdir(wavfile_dir)):
        feature = generate_feature(wavfile_dir + "/" + filename)
        label = generate_label_wavefiles(wavfile_dir + "/" + filename)
        label_name = label["emotion"] + "__" + label["gender"]
        data_row = [feature, label_name]
        data.append(data_row)
    print("[*] Generated features for directory: " + wavfile_dir)
    return data


def generate_dataset(dataset_path):
    data = []
    os.system("cls" if os.name == "nt" else "clear")
    print("[*] Generating features for dataset: " + dataset_path)
    for dir in tqdm(os.listdir(dataset_path)):
        dir_path = dataset_path + "/" + dir
        data += generate_dir_features(dir_path)
        os.system("cls" if os.name == "nt" else "clear")
        print("[*] Generating features for dataset: " + dataset_path)
    print("[*] Generated features for dataset: " + dataset_path)

    dataset = pd.DataFrame(data, columns=["feature", "class_label"])
    dataset = dataset.fillna(0)
    dataset = shuffle(dataset)

    return dataset


WAVE_PATH = "dataset/Actor_01/03-01-01-01-01-01-01.wav"

features = generate_dataset("dataset")
features = pd.DataFrame(features, columns=["feature", "class_label"])
print(features)

# feature = generate_feature(WAVE_PATH)
# print(feature)

# sampling_rate, x = scipy.io.wavfile.read(WAVE_PATH)
# image = generate_spectrogram(sampling_rate=sampling_rate, x=x)

# plt.imshow(image, interpolation="nearest", origin="lower", aspect="auto")

# plt.show()
