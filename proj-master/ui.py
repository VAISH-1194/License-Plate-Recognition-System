import tkinter as tk
from tkinter import filedialog, Label, Button
import cv2
from PIL import Image, ImageTk
import numpy as np
import pytesseract

# Set Tesseract Path (Modify as per your installation)
pytesseract.pytesseract.tesseract_cmd = r"F:\Tesseract\tesseract.exe"


# Function to process the uploaded image
def process_image():
    global panel

    # Open file dialog to select an image
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    
    if not file_path:
        return  # If no file selected, do nothing

    # Read the image
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    # Apply edge detection and contour detection
    edged = cv2.Canny(gray, 170, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (largest to smallest) and extract possible plate
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    plate = None

    for contour in contours:
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        if len(approx) == 4:  # License plates are rectangular
            plate = approx
            break

    if plate is not None:
        x, y, w, h = cv2.boundingRect(plate)
        plate_img = gray[y:y+h, x:x+w]  # Extract the plate region
        
        # Use Tesseract OCR to extract text
        plate_text = pytesseract.image_to_string(plate_img, config='--psm 8')
        result_label.config(text=f"Detected Plate: {plate_text.strip()}")

        # Draw the detected plate on the image
        cv2.drawContours(image, [plate], -1, (0, 255, 0), 3)
    else:
        result_label.config(text="No license plate detected!")

    # Convert image to display in Tkinter
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image = image.resize((400, 300), Image.Resampling.LANCZOS)
    image_tk = ImageTk.PhotoImage(image)

    # Update panel with new image
    panel.config(image=image_tk)
    panel.image = image_tk

# Create Tkinter Window
root = tk.Tk()
root.title("License Plate Recognition")
root.geometry("500x500")

# Upload Button
upload_btn = Button(root, text="Upload Image", command=process_image, font=("Arial", 14))
upload_btn.pack(pady=20)

# Image Display Panel
panel = Label(root)
panel.pack()

# Label for Result
result_label = Label(root, text="Upload an image to detect the plate!", font=("Arial", 12))
result_label.pack(pady=20)

# Run Tkinter main loop
root.mainloop()
