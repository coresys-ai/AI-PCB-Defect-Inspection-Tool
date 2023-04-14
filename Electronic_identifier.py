import cv2
import pytesseract
import streamlit as st
import numpy as np

def identify_components(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to the image
    threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    # Find contours in the image
    contours, hierarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop over the contours to identify the electronic components
    for contour in contours:
        # Get the x, y, width, and height of the contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Crop the contour from the image
        component = image[y:y+h, x:x+w]
        
        # Apply OCR to the cropped image
        text = pytesseract.image_to_string(component, config='--psm 10')
        
        # Display the component and its text
        st.image(component)
        st.write(text)

def main():
    st.title('Electronic Component Identifier')
    
