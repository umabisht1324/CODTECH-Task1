import streamlit as st
from keras.preprocessing import image as keras_image
from keras.models import load_model as keras_load_model
import numpy as np
from keras.applications.vgg16 import preprocess_input

def load_keras_model():
    model = keras_load_model("best_model2.h5")
    return model

model = load_keras_model()

def predict_image_class(image_path, model, class_folders):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = keras_image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    processed_img = preprocess_input(img_array)
    predictions = model.predict(processed_img)
    predicted_class = np.argmax(predictions)
    return class_folders[predicted_class]

def main():
    st.title("Lung Cancer Detection")
    st.write("Upload an image of lung tissue to predict the type of lung cancer.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = keras_image.load_img(uploaded_file, target_size=(224, 224))
        st.image(image, caption='Uploaded Image', use_column_width=True)
        
        st.write("")
        st.write("Classifying...")
         
        class_folders = ['lung_aca', 'lung_n', 'lung_scc']
        predicted_class = predict_image_class(uploaded_file, model, class_folders)
        st.write(f"Predicted class: {predicted_class}")

if __name__ == '__main__':
    main()
