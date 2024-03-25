import streamlit as st
import os
import helmet_detection
import cv2

def save_file(file, folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    with open(os.path.join(folder_path, file.name), "wb") as f:
        f.write(file.getbuffer())
    return os.path.join(folder_path, file.name)

def pred_image(img_path):
    frame = cv2.imread(img_path)
    out_img,result = helmet_detection.predict_and_detect(frame)
    out_img_path = "out_image.jpg"
    cv2.imwrite(out_img_path, out_img)
    return out_img_path

def pred_vid(vid_path):
    out_vid_path,res = helmet_detection.get_vid_predictions(vid_path)
    print(out_vid_path)
    return out_vid_path

def main():

    st.title("Detection App")
    sub_option = st.selectbox("Choose detection method:", ("Image Prediction", "Video Prediction"))

    if sub_option == "Image Prediction":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image_path = save_file(uploaded_file, "user_input")
            if st.button('Predict', key='predict_button'):
                with st.spinner('Processing image...'):
                    result_path = pred_image(image_path)
                    st.image(result_path, caption="Predicted Image")

    else:
        uploaded_file = st.file_uploader("Upload Video", type=["mp4"])
        if uploaded_file is not None:
            video_path = save_file(uploaded_file, "user_input")
            if st.button('Predict', key='predict_button'):
                with st.spinner('Processing video...'):
                    result_path = pred_vid(video_path)
                    st.video(result_path, format="mp4")

    # Add custom CSS for styled button
    predict_button_style = """
        <style>
            .predict-button {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 15px 32px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
            }
        </style>
    """
    st.markdown(predict_button_style, unsafe_allow_html=True)

if __name__ == "__main__":
    main()


