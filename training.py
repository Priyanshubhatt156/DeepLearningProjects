import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Flatten, TimeDistributed, Dropout
import cv2
import numpy as np
import os

# --- 1. MODEL DEFINITION ---
def build_cnn_lstm(seq_len=10, img_h=158, img_w=238):
    model = Sequential([
        # Extract features from each frame in the sequence
        TimeDistributed(Conv2D(32, (3, 3), activation='relu'), input_shape=(seq_len, img_h, img_w, 1)),
        TimeDistributed(MaxPooling2D((2, 2))),
        TimeDistributed(Flatten()),
        
        # Analyze the sequence of features
        LSTM(64, return_sequences=False),
        Dropout(0.5),
        
        # Output a single number (the count)
        Dense(1, activation='linear') 
    ])
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- 2. DATA LOADING LOGIC ---
def load_training_data(dataset_path, seq_len=10):
    X_train = []
    Y_train = []
    
    # FIX: Using absolute path or moving up one level to find the Data folder
    # This assumes your Data folder is in the root 'DeepLearningProjects'
    base_dir = os.path.dirname(os.path.abspath(__file__))
    train_folder = os.path.join(base_dir, '..', 'Data', 'UCSD_Dataset', 'UCSDped1', 'Train', 'Train001')
    
    if not os.path.exists(train_folder):
        print(f"❌ Error: Folder NOT found at: {os.path.abspath(train_folder)}")
        print("Check if 'Data' folder is in the root directory.")
        return None, None

    images = sorted([img for img in os.listdir(train_folder) if img.endswith('.tif')])
    
    if len(images) < seq_len:
        print("❌ Error: Not enough images in folder to create a sequence.")
        return None, None

    for i in range(min(len(images) - seq_len, 50)): # Limiting to 50 samples for speed
        sequence = []
        for j in range(seq_len):
            img_path = os.path.join(train_folder, images[i + j])
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (238, 158)) 
            sequence.append(img / 255.0)      
        
        X_train.append(sequence)
        Y_train.append(15) # Example label (15 people)

    return np.expand_dims(np.array(X_train), -1), np.array(Y_train)

# --- 3. EXECUTION ---
if __name__ == "__main__":
    print("🚀 Loading data...")
    x, y = load_training_data('Data/UCSD_Dataset')
    
    if x is not None:
        print("🧠 Initializing Model...")
        model = build_cnn_lstm()
        
        print("⏳ Starting Training (Epochs 1-5)...")
        model.fit(x, y, epochs=5, batch_size=2)
        
        # --- 4. SAVING ---
        # Save it to the PARENT folder so Apps/routes.py can see it immediately
        save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'crowd_model.h5')
        model.save(save_path)
        print(f"✅ SUCCESS: Model saved at {os.path.abspath(save_path)}")
    else:
        print("❌ Training aborted because data was not found.")