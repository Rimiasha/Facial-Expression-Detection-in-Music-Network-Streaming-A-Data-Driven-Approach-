import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase, WebRtcMode
import random
from streamlit_player import st_player
import cv2
import numpy as np
from keras.models import model_from_json
import av
import queue



model_path = './model/'
img_size = 48
emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
song_path = "./music/"
pixel_format = "bgr24"
num_class = len(emotion_labels)

json_file = open(model_path + 'model_json.json')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(model_path + 'model_weight.h5')
cascade = cv2.CascadeClassifier(model_path + 'haarcascade_frontalface_default.xml')
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

emotion_list = []
proba = queue.Queue()

def video_frame_callback(frame):
    frm = frame.to_ndarray(format=pixel_format)
    gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
    faceLands = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=1, minSize=(120, 120))
    if len(faceLands) > 0:
        for faceLand in faceLands:
            x, y, w, h = faceLand
            result = np.array([0.0] * num_class)
            image = cv2.resize(gray[y:y + h, x:x + w], (img_size, img_size))
            image = image / 255.0
            image = image.reshape(1, img_size, img_size, 1)
            predict_lists = model.predict(image, batch_size=32, verbose=1)

            result += np.array([predict for predict_list in predict_lists
                                for predict in predict_list])

            emotion = emotion_labels[int(np.argmax(result))]
            emotion_list.append(emotion)

            proba.put(predict_lists)


            cv2.rectangle(frm, (x - 20, y - 20), (x + w + 20, y + h + 20),
                          (0, 255, 255), thickness=5)
            cv2.putText(frm, '%s' % emotion, (x, y - 50), cv2.FONT_ITALIC, 1, (0, 0, 255), 2, cv2.LINE_AA,False)

    return av.VideoFrame.from_ndarray(frm, format=pixel_format)


st.set_page_config(
    page_icon="🎸",
    page_title="Music Network Streaming",
    layout="centered",
)


def main():
    st.title("Welcome to Music Network Streaming!!")
    stop_warn = st.empty()
    ctx = webrtc_streamer(key="youmustnotpasslol", video_frame_callback=video_frame_callback,
                          rtc_configuration=RTC_CONFIGURATION, media_stream_constraints={"audio": False, "video": True})
    st.markdown("###### Let's stream some music!")

    emotion_songs = {}  # dictionary to hold songs for each emotion in the playlist
    for emotion in emotion_labels:
        with open(f'{song_path}{emotion}.txt') as f:
            emotion_songs[emotion] = [line.strip() for line in f.readlines()]

    while ctx.state.playing:
        if not emotion_list:
            continue

        final_emotion = max(set(emotion_list), key=emotion_list.count)
        st.write(f"Final Emotion Detected: {final_emotion}")

        # Play all songs from the playlist of the predicted emotion
        if final_emotion in emotion_songs:
            with open(f'{song_path}{final_emotion}.txt') as f:
                songs = f.readlines()

            for song in songs:
                st_player(song, key=song)
        else:
            st.write(f"No songs available for {final_emotion} emotion.")

        stop_warn.warning("Press 'STOP' to reset streaming")
        break

    st.markdown("---")

page_names_to_func = {
    "Welcome to Music Network Streaming!!": main
}
page_names_to_func["Welcome to Music Network Streaming!!"]()
