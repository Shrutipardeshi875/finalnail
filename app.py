import os
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage import io
from skimage.transform import resize
import autokeras as ak  # Import AutoKeras

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flashing messages

# Path for data storage
APPOINTMENTS_FILE = 'appointments.csv'
UPLOAD_FOLDER = 'uploads'  # Folder to store uploaded images
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Parameters for the model
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 32
EPOCHS = 20

# Load existing appointments or create a new DataFrame
if os.path.exists(APPOINTMENTS_FILE):
    appointments_df = pd.read_csv(APPOINTMENTS_FILE)
else:
    appointments_df = pd.DataFrame(columns=['Name', 'Date', 'Symptoms', 'Medical History', 'Hemoglobin Count'])

# Load and prepare images for PCA
def load_and_preprocess_images(data_dir):
    image_data = []
    labels = []

    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        if os.path.isdir(class_dir):
            for img_file in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_file)
                img = io.imread(img_path)
                if img.ndim == 3:  # Ensure it's a 3D image
                    img_resized = resize(img, (IMG_WIDTH, IMG_HEIGHT), anti_aliasing=True)
                    image_data.append(img_resized / 255.0)  # Keep the 3D structure of the image (height, width, channels)
                    labels.append(label)

    return np.array(image_data), np.array(labels)

# Load data for AutoKeras
def prepare_data(data_dir):
    image_data, labels = load_and_preprocess_images(data_dir)
    encoder = OneHotEncoder(sparse=False)
    labels_encoded = encoder.fit_transform(labels.reshape(-1, 1))

    return image_data, labels_encoded

# Home Route
@app.route('/')
def index():
    return render_template('index.html')

# Appointment Route
@app.route('/appointment', methods=['GET', 'POST'])
def appointment():
    if request.method == 'POST':
        name = request.form['name']
        date = request.form['date']
        symptoms = request.form['symptoms']
        medical_history = request.form['medical_history']
        hemoglobin_count = request.form['hemoglobin_count']

        # Save appointment data to CSV
        appointment_data = {
            'Name': name,
            'Date': date,
            'Symptoms': symptoms,
            'Medical History': medical_history,
            'Hemoglobin Count': hemoglobin_count
        }

        # Append new appointment
        global appointments_df
        appointments_df = appointments_df.append(appointment_data, ignore_index=True)
        appointments_df.to_csv(APPOINTMENTS_FILE, index=False)

        return redirect(url_for('index'))  # Redirect back to the home page

    return render_template('appointment.html')

# Upload Dataset Route
@app.route('/upload', methods=['GET', 'POST'])
def upload_dataset():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        
        files = request.files.getlist('file')
        for file in files:
            if file.filename == '':
                flash('No selected file')
                return redirect(request.url)
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

        # Inform the user about the successful upload
        flash('Files uploaded successfully!')
        return redirect(url_for('index'))

    return render_template('upload.html')

# Model Training Route
@app.route('/train_model', methods=['GET', 'POST'])
def train_model():
    # Prepare data from the uploaded dataset
    image_data, labels_encoded = prepare_data(UPLOAD_FOLDER)

    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(image_data, labels_encoded, test_size=0.2, random_state=42)

    # Build and train the AutoKeras model
    clf = ak.ImageClassifier(
        overwrite=True,
        max_trials=3  # Number of different models to try
    )

    # Train the model
    clf.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val))

    # Evaluate the model
    accuracy = clf.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {accuracy}")

    # Export the best model
    model = clf.export_model()
    model.save('autokeras_nail_disease_model.h5')  # Save the best AutoKeras model

    return f'Model trained with AutoKeras and saved successfully! Validation Accuracy: {accuracy[1]}'

if __name__ == '__main__':
    app.run(debug=True)
