import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/pcb_1st/weights/best.pt')

# Define the inference function
def detect_pcb(image):
    # Convert the image to a numpy array
    img = np.array(image.convert('RGB'))

    # Perform inference on the image
    results = model(img)

    # Get the predicted bounding boxes and labels
    bboxes = results.xyxy[0].cpu().numpy()
    labels = results.names[results.xyxy[0][:, -1].long().cpu()]

    # Draw the predicted bounding boxes and labels on the image
    for bbox, label in zip(bboxes, labels):
        x1, y1, x2, y2 = bbox.astype(np.int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Convert the image back to a PIL image
    output_image = Image.fromarray(img)

    return output_image

# Define the Streamlit app
def app():
    # Set the app title
    st.title('YOLOv5 PCB Detection')

    # Create an input file uploader
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    # Perform inference when the user uploads an image
    if uploaded_file is not None:
        # Read the image
        image = Image.open(uploaded_file)

        # Perform object detection using the YOLOv5 model
        output_image = detect_pcb(image)

        # Display the output image with the predicted bounding boxes and labels
        st.image(output_image, caption='Output', use_column_width=True)

# Run the app
if __name__ == '__main__':
    app()
