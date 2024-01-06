import speech_recognition as sr


class SpeechToText:
    def __init__(self):
        self.recognizer = sr.Recognizer()

    def recognize(self, audio_path):
        audio = sr.AudioFile(audio_path)
        with audio as source:
            audio = self.recognizer.record(source)
            text = self.recognizer.recognize_google(audio)
            return text


# recognizer = SpeechToText()
# result = recognizer.recognize("sample.wav")
# print(result)
