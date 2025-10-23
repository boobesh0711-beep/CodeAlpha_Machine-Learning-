# -----------------------------------------------------------
# TASK 2: Emotion Recognition from Speech
# CodeAlpha Internship – Machine Learning Project
# -----------------------------------------------------------

# Step 1: Import Libraries
import os
import librosa
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# -----------------------------------------------------------
# Step 2: Dataset Loading
# -----------------------------------------------------------

# Example path (Change this to your dataset folder path)
data_path = '/content/ravdess-data/'  # Change to your directory

# Emotion labels for RAVDESS dataset
emotion_map = {
    '01': 'neutral',
    '02': 'calm',
    '03': 'happy',
    '04': 'sad',
    '05': 'angry',
    '06': 'fearful',
    '07': 'disgust',
    '08': 'surprised'
}

def extract_features(file_path):
    """Extract MFCC features from an audio file."""
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfccs_scaled = np.mean(mfccs.T, axis=0)
    return mfccs_scaled

features = []
labels = []

# -----------------------------------------------------------
# Step 3: Feature Extraction
# -----------------------------------------------------------

for root, dirs, files in os.walk(data_path):
    for file in files:
        if file.endswith('.wav'):
            file_path = os.path.join(root, file)
            # Extract emotion code
            emotion_code = file.split('-')[2]
            emotion = emotion_map.get(emotion_code)
            if emotion:
                feature = extract_features(file_path)
                features.append(feature)
                labels.append(emotion)

print("✅ Feature extraction completed!")
print("Total samples:", len(features))

# -----------------------------------------------------------
# Step 4: Data Preparation
# -----------------------------------------------------------

X = np.array(features)
y = np.array(labels)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

# Reshape input for CNN (samples, height, width, channels)
X_train = X_train.reshape(X_train.shape[0], 40, 1, 1)
X_test = X_test.reshape(X_test.shape[0], 40, 1, 1)

print("✅ Data ready for model training!")

# -----------------------------------------------------------
# Step 5: CNN Model Building
# -----------------------------------------------------------

model = Sequential([
    Conv2D(64, kernel_size=(2,2), activation='relu', input_shape=(40,1,1)),
    MaxPooling2D(pool_size=(2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_categorical.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -----------------------------------------------------------
# Step 6: Train the Model
# -----------------------------------------------------------

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop]
)

# -----------------------------------------------------------
# Step 7: Evaluate the Model
# -----------------------------------------------------------

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"\n🎯 Test Accuracy: {test_acc*100:.2f}%")

# -----------------------------------------------------------
# Step 8: Visualization
# -----------------------------------------------------------

plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# -----------------------------------------------------------
# Step 9: Save the Model
# -----------------------------------------------------------

model.save('emotion_recognition_model.h5')
print("💾 Model saved as emotion_recognition_model.h5")

# -----------------------------------------------------------
# Step 10: Prediction Example
# -----------------------------------------------------------

def predict_emotion(file_path):
    feature = extract_features(file_path)
    feature = feature.reshape(1, 40, 1, 1)
    prediction = model.predict(feature)
    predicted_label = encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

# Example
# test_file = '/content/ravdess-data/Actor_01/03-01-03-01-01-01-01.wav'
# print("Predicted Emotion:", predict_emotion(test_file))
