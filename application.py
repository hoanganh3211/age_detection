import cv2
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np


@st.cache_resource
def load_keras_model(path):
    """Tải model Keras."""
    try:
        model = tf.keras.models.load_model(path)
        return model
    except Exception as e:
        st.error(f"Error loading Keras model: {e}")
        return None


@st.cache_resource
def load_haarcascade(path):
    """Tải file Haar Cascade."""
    try:
        face_cascade = cv2.CascadeClassifier(path)
        return face_cascade
    except Exception as e:
        st.error(f"Lỗi khi tải Haar Cascade: {e}")
        return None


def main():
    st.header("How old are you according to a CNN")
    st.write(
        "Upload an image of yourself below to find out!")

    model_path = "D:\\GIT\\age_detection\\model\\agemodel.h5"

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

    model = load_keras_model(model_path)
    face_cascade = load_haarcascade(cascade_path)

    file = st.file_uploader("Upload Photo")

    if file is not None:
        pil_image = Image.open(file).convert('RGB')
        st.image(pil_image, caption="Original Image", width=400)

        open_cv_image = np.array(pil_image)
        open_cv_image = open_cv_image[:, :, ::-1].copy()

        gray_image = cv2.cvtColor(open_cv_image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        if len(faces) == 0:
            st.warning("Don't see any faces in the image. Please try another one.")

        elif len(faces) == 1:

            x, y, w, h = faces[0]

            # Cắt khuôn mặt từ ảnh gốc (ảnh màu)
            padding = 15
            face_cropped = open_cv_image[max(0, y - padding):min(y + h + padding, open_cv_image.shape[0]),
                               max(0, x - padding):min(x + w + padding, open_cv_image.shape[1])]

            face_rgb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)

            image = tf.image.resize(face_rgb, [200, 200])
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = image / 255.0
            image = tf.expand_dims(image, axis=0)
            age = model.predict(image)
            predicted_age_value = int(age[0][0])

            st.markdown(f"## You're {predicted_age_value} years old according to our CNN!")

        else:
            st.success(f"Found {len(faces)} face(s) in the image.")

            for i, (x, y, w, h) in enumerate(faces):
                st.write(f"---")
                st.subheader(f"Face {i + 1}")

                # Cắt khuôn mặt từ ảnh gốc (ảnh màu)
                padding = 15
                face_cropped = open_cv_image[max(0, y - padding):min(y + h + padding, open_cv_image.shape[0]),
                               max(0, x - padding):min(x + w + padding, open_cv_image.shape[1])]

                face_rgb = cv2.cvtColor(face_cropped, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns([1, 3])

                with col1:
                    st.image(face_rgb, width=120)

                with col2:
                    image = tf.image.resize(face_rgb, [200, 200])
                    image = tf.keras.preprocessing.image.img_to_array(image)
                    image = image / 255.0
                    image = tf.expand_dims(image, axis=0)  # Thêm batch dimension

                    age = model.predict(image)
                    predicted_age_value = int(age[0][0])
                    st.markdown(f"## Predicted age: {predicted_age_value}")


if __name__ == '__main__':
    main()