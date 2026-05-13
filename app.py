# app.py
import streamlit as st
import cv2
import pickle
import numpy as np
import os
import pandas as pd
from datetime import datetime
import time
from sklearn.neighbors import KNeighborsClassifier
import warnings
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Face Recognition Attendance System",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
    .stButton button {
        width: 100%;
        border-radius: 0.5rem;
        height: 3rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'face_data' not in st.session_state:
    st.session_state.face_data = np.array([])
if 'names' not in st.session_state:
    st.session_state.names = []
if 'attendance_data' not in st.session_state:
    st.session_state.attendance_data = []
if 'knn_model' not in st.session_state:
    st.session_state.knn_model = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'color_mode' not in st.session_state:
    st.session_state.color_mode = "color"  # Default to color

# Create necessary directories
def create_directories():
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('attendance'):
        os.makedirs('attendance')

create_directories()

# Get expected feature dimensions based on color mode
def get_expected_features():
    if st.session_state.color_mode == "grayscale":
        return 50 * 50  # 2500 features for grayscale
    else:
        return 50 * 50 * 3  # 7500 features for color

# Load or initialize face data with consistent dimensions
def load_face_data():
    try:
        if os.path.exists('data/faces_data.pkl'):
            with open('data/faces_data.pkl', 'rb') as f:
                loaded_data = pickle.load(f)
                
            if len(loaded_data) > 0:
                st.session_state.face_data = loaded_data
                # Detect color mode from existing data
                if loaded_data[0].shape[0] == 2500:
                    st.session_state.color_mode = "grayscale"
                else:
                    st.session_state.color_mode = "color"
            else:
                st.session_state.face_data = np.array([])
        else:
            st.session_state.face_data = np.array([])
            
        if os.path.exists('data/names.pkl'):
            with open('data/names.pkl', 'rb') as f:
                st.session_state.names = pickle.load(f)
        else:
            st.session_state.names = []
            
    except Exception as e:
        st.error(f"Error loading face data: {e}")
        st.session_state.face_data = np.array([])
        st.session_state.names = []

load_face_data()

# Preprocess face image to ensure consistent dimensions
def preprocess_face(face_img, target_shape=(50, 50)):
    """
    Preprocess face image to ensure consistent dimensions based on color mode
    """
    # Resize to target shape
    resized = cv2.resize(face_img, target_shape)
    
    # Convert based on color mode
    if st.session_state.color_mode == "grayscale":
        if len(resized.shape) == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    else:  # color mode
        if len(resized.shape) == 2:  # If grayscale, convert to color
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)
    
    return resized

# Convert existing data to new color mode if needed
def convert_existing_data(new_color_mode):
    if len(st.session_state.face_data) == 0:
        return True
        
    if st.session_state.color_mode == new_color_mode:
        return True
        
    st.warning(f"Converting existing data from {st.session_state.color_mode} to {new_color_mode} mode...")
    
    try:
        converted_data = []
        for face in st.session_state.face_data:
            # Reshape back to image dimensions
            if st.session_state.color_mode == "grayscale":
                # Current: grayscale (2500) -> Need to convert to color (7500)
                img = face.reshape(50, 50)
                img_color = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
                converted_face = img_color.flatten()
            else:
                # Current: color (7500) -> Need to convert to grayscale (2500)
                img = face.reshape(50, 50, 3)
                img_gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
                converted_face = img_gray.flatten()
            
            converted_data.append(converted_face)
        
        st.session_state.face_data = np.array(converted_data)
        st.session_state.color_mode = new_color_mode
        st.success("Data conversion completed!")
        return True
        
    except Exception as e:
        st.error(f"Error converting data: {e}")
        return False

# Train or load KNN model
def train_knn_model():
    if len(st.session_state.face_data) > 0 and len(st.session_state.names) > 0:
        try:
            # Verify all data has consistent dimensions
            expected_features = get_expected_features()
            valid_indices = []
            
            for i, face in enumerate(st.session_state.face_data):
                if face.shape[0] == expected_features:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                st.error("No valid training data found with consistent dimensions.")
                return False
                
            valid_data = st.session_state.face_data[valid_indices]
            valid_names = [st.session_state.names[i] for i in valid_indices]
            
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(valid_data, valid_names)
            st.session_state.knn_model = knn
            st.session_state.model_trained = True
            return True
        except Exception as e:
            st.error(f"Error training model: {e}")
            return False
    else:
        st.warning("No face data available for training. Please add faces first.")
        return False

# Save face data to files
def save_face_data():
    try:
        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(st.session_state.face_data, f)
        
        with open('data/names.pkl', 'wb') as f:
            pickle.dump(st.session_state.names, f)
        
        # Save color mode
        with open('data/color_mode.pkl', 'wb') as f:
            pickle.dump(st.session_state.color_mode, f)
            
        return True
    except Exception as e:
        st.error(f"Error saving face data: {e}")
        return False

# Load color mode
try:
    if os.path.exists('data/color_mode.pkl'):
        with open('data/color_mode.pkl', 'rb') as f:
            st.session_state.color_mode = pickle.load(f)
except:
    pass

# Main app header
st.markdown('<h1 class="main-header">👤 Face Recognition Attendance System</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose a module", 
                               ["Home", "Add New Face", "Take Attendance", "View Attendance Records", "Manage Data"])

# Home page
if app_mode == "Home":
    st.markdown('<h2 class="sub-header">Welcome to the Face Recognition Attendance System</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("""
        This application allows you to:
        - Register new faces for attendance tracking
        - Take attendance using face recognition
        - View and manage attendance records
        - Export attendance data
        
        ### How to use:
        1. **Add New Face**: Register individuals by capturing their facial data
        2. **Take Attendance**: Use your camera to recognize faces and mark attendance
        3. **View Records**: Check attendance history and export data
        4. **Manage Data**: View or delete registered face data
        """)
    
    with col2:
        st.markdown("<div style='text-align: center; font-size: 100px;'>👤</div>", unsafe_allow_html=True)
    
    # Statistics
    st.markdown('<h3 class="sub-header">System Statistics</h3>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Registered Faces", len(set(st.session_state.names)) if st.session_state.names else 0)
    
    with col2:
        attendance_files = [f for f in os.listdir('attendance') if f.endswith('.csv')] if os.path.exists('attendance') else []
        st.metric("Attendance Records", len(attendance_files))
    
    with col3:
        if st.session_state.model_trained:
            st.metric("Model Status", "Trained")
        else:
            st.metric("Model Status", "Not Trained")
    
    with col4:
        st.metric("Color Mode", st.session_state.color_mode.capitalize())
    
    # Quick actions
    st.markdown('<h3 class="sub-header">Quick Actions</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Train Recognition Model", use_container_width=True):
            if train_knn_model():
                st.success("Model trained successfully!")
            else:
                st.error("Failed to train model. Check if you have registered faces.")
    
    with col2:
        if st.button("Check System Status", use_container_width=True):
            if st.session_state.model_trained:
                st.success("System is ready for attendance tracking!")
            else:
                st.warning("System needs training. Please register faces and train the model.")

# Add New Face page
elif app_mode == "Add New Face":
    st.markdown('<h2 class="sub-header">Register New Face</h2>', unsafe_allow_html=True)
    
    # Color mode selection with conversion warning
    new_color_mode = st.radio("Select Color Mode:", 
                             ["color", "grayscale"], 
                             index=0 if st.session_state.color_mode == "color" else 1,
                             format_func=lambda x: "Color (3 channels)" if x == "color" else "Grayscale (1 channel)")
    
    if new_color_mode != st.session_state.color_mode and len(st.session_state.face_data) > 0:
        st.warning(f"Changing color mode will convert existing {len(st.session_state.face_data)} face samples.")
        if st.button("Convert Existing Data"):
            if convert_existing_data(new_color_mode):
                save_face_data()
    
    st.session_state.color_mode = new_color_mode
    expected_features = get_expected_features()
    st.info(f"Current mode: {st.session_state.color_mode.upper()} | Expected features: {expected_features}")
    
    name = st.text_input("Enter Name for Registration")
    
    if name:
        st.info(f"Registering face for: {name}")
        
        # Initialize camera
        run_camera = st.checkbox("Start Camera for Face Capture")
        
        if run_camera:
            # Create a placeholder for the camera feed
            camera_placeholder = st.empty()
            
            # Load face detector
            try:
                face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                facedetect = cv2.CascadeClassifier(face_cascade_path)
                
                if facedetect.empty():
                    st.error("Failed to load face detector cascade file.")
                    st.stop()
            except Exception as e:
                st.error(f"Error loading face detector: {e}")
                st.stop()
            
            # Initialize variables for face capture
            if 'face_capture_data' not in st.session_state:
                st.session_state.face_capture_data = []
                st.session_state.face_capture_count = 0
            
            # Camera capture loop
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access camera. Please check if camera is available.")
                st.stop()
            
            stop_camera = st.button("Stop Camera")
            
            while run_camera and not stop_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)
                
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                    
                    # Capture face data every 10 frames
                    if len(st.session_state.face_capture_data) < 20 and st.session_state.face_capture_count % 10 == 0:
                        crop_img = frame[y:y + h, x:x + w]
                        processed_face = preprocess_face(crop_img)
                        flattened = processed_face.flatten()
                        
                        # Verify dimensions
                        if flattened.shape[0] == expected_features:
                            st.session_state.face_capture_data.append(flattened)
                        else:
                            st.error(f"Unexpected feature size: {flattened.shape[0]} (expected {expected_features})")
                    
                    st.session_state.face_capture_count += 1
                    cv2.putText(frame, f"Captured: {len(st.session_state.face_capture_data)}/20", 
                               (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 2)
                
                # Display the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB")
                
                # Check if we've captured enough faces
                if len(st.session_state.face_capture_data) >= 20:
                    st.success("Successfully captured 20 face samples!")
                    break
                
                # Small delay
                time.sleep(0.1)
            
            # Release camera
            cap.release()
            
            # Save captured face data
            if len(st.session_state.face_capture_data) > 0:
                if st.button("Save Face Data"):
                    # Convert to numpy array
                    faces_array = np.array(st.session_state.face_capture_data)
                    
                    # Update face data and names
                    if st.session_state.face_data.size == 0:
                        st.session_state.face_data = faces_array
                        st.session_state.names = [name] * len(st.session_state.face_capture_data)
                    else:
                        # Verify dimensions match before concatenating
                        if st.session_state.face_data.shape[1] == faces_array.shape[1]:
                            st.session_state.face_data = np.vstack([st.session_state.face_data, faces_array])
                            st.session_state.names.extend([name] * len(st.session_state.face_capture_data))
                        else:
                            st.error(f"Dimension mismatch! Cannot add new data.")
                            st.info(f"Existing: {st.session_state.face_data.shape[1]} features, New: {faces_array.shape[1]} features")
                            st.stop()
                    
                    # Save to files
                    if save_face_data():
                        st.success(f"Successfully registered {name} with {len(st.session_state.face_capture_data)} face samples!")
                        
                        # Reset capture data
                        st.session_state.face_capture_data = []
                        st.session_state.face_capture_count = 0
                        
                        # Retrain model
                        train_knn_model()
                    else:
                        st.error("Failed to save face data.")

# Take Attendance page
elif app_mode == "Take Attendance":
    st.markdown('<h2 class="sub-header">Take Attendance</h2>', unsafe_allow_html=True)
    
    # Check if model is trained
    if not st.session_state.model_trained:
        st.warning("Please train the recognition model first from the Home page.")
        if st.button("Train Model Now"):
            if train_knn_model():
                st.success("Model trained successfully!")
            else:
                st.error("Failed to train model. Please register faces first.")
        st.stop()
    
    subject = st.text_input("Enter Subject Name", placeholder="e.g., Mathematics, Physics")
    
    if subject:
        st.info(f"Taking attendance for: {subject}")
        st.info(f"Current mode: {st.session_state.color_mode.upper()} | Model expects {get_expected_features()} features")
        
        # Initialize camera for attendance
        run_attendance = st.checkbox("Start Camera for Attendance")
        
        if run_attendance:
            # Create placeholders
            camera_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Load face detector
            try:
                face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                facedetect = cv2.CascadeClassifier(face_cascade_path)
                
                if facedetect.empty():
                    st.error("Failed to load face detector cascade file.")
                    st.stop()
            except Exception as e:
                st.error(f"Error loading face detector: {e}")
                st.stop()
            
            # Initialize variables
            recognized_faces = set()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Cannot access camera. Please check if camera is available.")
                st.stop()
                
            stop_camera = st.button("Stop Camera")
            
            while run_attendance and not stop_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame from camera")
                    break
                
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = facedetect.detectMultiScale(gray, 1.3, 5)
                
                current_recognitions = []
                
                for (x, y, w, h) in faces:
                    try:
                        # Extract and preprocess face
                        crop_img = frame[y:y + h, x:x + w]
                        processed_face = preprocess_face(crop_img)
                        flattened = processed_face.flatten().reshape(1, -1)
                        
                        # Check if dimensions match the model
                        if flattened.shape[1] != get_expected_features():
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, f"Dim Error: {flattened.shape[1]}", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            continue
                            
                        prediction = st.session_state.knn_model.predict(flattened)
                        confidence = np.max(st.session_state.knn_model.predict_proba(flattened))
                        
                        # Only accept predictions with reasonable confidence
                        if confidence > 0.6:
                            person_name = prediction[0]
                            current_recognitions.append(person_name)
                            
                            # Draw rectangle and name
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
                            cv2.putText(frame, f"{person_name} ({confidence:.2f})", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 255), 2)
                            
                            # Add to recognized faces if not already there
                            if person_name not in recognized_faces:
                                recognized_faces.add(person_name)
                                
                                # Record attendance
                                ts = time.time()
                                date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
                                timestamp = datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                                
                                attendance_record = {
                                    'Name': person_name,
                                    'Subject': subject,
                                    'Date': date,
                                    'Time': timestamp
                                }
                                
                                # Save to CSV
                                csv_file = f"attendance/ATTENDANCE_{date}.csv"
                                df = pd.DataFrame([attendance_record])
                                
                                if os.path.exists(csv_file):
                                    existing_df = pd.read_csv(csv_file)
                                    # Check if this person already has attendance for this subject today
                                    mask = (existing_df['Name'] == person_name) & (existing_df['Subject'] == subject) & (existing_df['Date'] == date)
                                    if not existing_df[mask].empty:
                                        status_placeholder.warning(f"{person_name} already marked for {subject} today")
                                    else:
                                        updated_df = pd.concat([existing_df, df], ignore_index=True)
                                        updated_df.to_csv(csv_file, index=False)
                                        status_placeholder.success(f"Attendance recorded for {person_name} at {timestamp}")
                                else:
                                    df.to_csv(csv_file, index=False)
                                    status_placeholder.success(f"Attendance recorded for {person_name} at {timestamp}")
                        else:
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, "Unknown", (x, y-10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    except Exception as e:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                        cv2.putText(frame, "Error", (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display the frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB")
                
                # Display current recognitions
                if current_recognitions:
                    st.sidebar.markdown("### Currently Detected:")
                    for name in set(current_recognitions):
                        st.sidebar.write(f"✅ {name}")
                
                # Small delay
                time.sleep(0.1)
            
            # Release camera
            cap.release()
            
            # Show summary
            if recognized_faces:
                st.markdown("### Attendance Summary")
                st.write(f"Recorded attendance for {len(recognized_faces)} people:")
                for name in recognized_faces:
                    st.write(f"- {name}")

# View Attendance Records page - COMPLETE VERSION
elif app_mode == "View Attendance Records":
    st.markdown('<h2 class="sub-header">📊 View Attendance Records</h2>', unsafe_allow_html=True)
    
    # Get list of attendance files
    attendance_files = []
    if os.path.exists('attendance'):
        attendance_files = [f for f in os.listdir('attendance') if f.endswith('.csv')]
        attendance_files.sort(reverse=True)  # Sort by most recent first
    
    if not attendance_files:
        st.info("No attendance records found. Take some attendance first!")
        st.info("Go to 'Take Attendance' page to start recording attendance.")
        
        # Create sample data for testing
        if st.button("Create Sample Data for Testing"):
            sample_data = {
                'Name': ['John Doe', 'Jane Smith', 'Mike Johnson', 'John Doe', 'Sarah Wilson'],
                'Subject': ['Mathematics', 'Physics', 'Mathematics', 'Physics', 'Mathematics'],
                'Date': ['01-12-2024', '01-12-2024', '01-12-2024', '02-12-2024', '02-12-2024'],
                'Time': ['09:00:00', '10:30:00', '14:00:00', '09:15:00', '11:00:00']
            }
            
            sample_df = pd.DataFrame(sample_data)
            sample_file = "attendance/ATTENDANCE_01-12-2024.csv"
            sample_df.to_csv(sample_file, index=False)
            st.success("Created sample attendance data!")
            st.rerun()
    else:
        # File selection
        selected_file = st.selectbox("Select Attendance File", attendance_files)
        
        if selected_file:
            try:
                # Load and display attendance data
                file_path = os.path.join('attendance', selected_file)
                df = pd.read_csv(file_path)
                
                if df.empty:
                    st.warning("Selected file is empty.")
                else:
                    # Display dataframe
                    st.markdown(f"### 📋 Attendance Records: {selected_file}")
                    st.dataframe(df, use_container_width=True)
                    
                    # Show statistics
                    st.markdown("### 📈 Statistics")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Records", len(df))
                    
                    with col2:
                        unique_people = df['Name'].nunique() if 'Name' in df.columns else 0
                        st.metric("Unique People", unique_people)
                    
                    with col3:
                        unique_subjects = df['Subject'].nunique() if 'Subject' in df.columns else 0
                        st.metric("Subjects", unique_subjects)
                    
                    with col4:
                        if 'Date' in df.columns:
                            dates = df['Date'].nunique()
                            st.metric("Dates", dates)
                    
                    # Show unique individuals
                    if 'Name' in df.columns:
                        st.markdown("### 👥 Individuals Present")
                        unique_names = df['Name'].unique()
                        for name in unique_names:
                            count = len(df[df['Name'] == name])
                            st.write(f"- **{name}**: {count} attendance record(s)")
                    
                    # Export options
                    st.markdown("### 💾 Export Data")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Download as CSV
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="Download as CSV",
                            data=csv,
                            file_name=selected_file,
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col2:
                        # Delete file option
                        if st.button("Delete This File", use_container_width=True, type="secondary"):
                            if st.checkbox("I'm sure I want to delete this attendance file"):
                                os.remove(file_path)
                                st.success(f"Deleted {selected_file}")
                                st.rerun()
            
            except Exception as e:
                st.error(f"Error loading attendance file: {e}")
                st.info("The file might be corrupted or in an unexpected format.")
        
        # Show all files summary
        st.markdown("### 📁 All Attendance Files Summary")
        if attendance_files:
            summary_data = []
            for file in attendance_files:
                try:
                    file_path = os.path.join('attendance', file)
                    df = pd.read_csv(file_path)
                    record_count = len(df)
                    date_str = file.replace('ATTENDANCE_', '').replace('.csv', '')
                    summary_data.append({
                        'File': file,
                        'Date': date_str,
                        'Records': record_count,
                        'People': df['Name'].nunique() if 'Name' in df.columns else 0
                    })
                except:
                    summary_data.append({
                        'File': file,
                        'Date': 'Error',
                        'Records': 0,
                        'People': 0
                    })
            
            summary_df = pd.DataFrame(summary_data)
            st.dataframe(summary_df, use_container_width=True)
            
            # Export all attendance data
            if st.button("Export All Attendance Data", use_container_width=True):
                try:
                    all_data = []
                    for file in attendance_files:
                        file_path = os.path.join('attendance', file)
                        df = pd.read_csv(file_path)
                        all_data.append(df)
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        csv_data = combined_df.to_csv(index=False)
                        
                        st.download_button(
                            label="Download All Attendance as CSV",
                            data=csv_data,
                            file_name="all_attendance_records.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                except Exception as e:
                    st.error(f"Error combining attendance data: {e}")
        else:
            st.info("No attendance files available.")

# Manage Data page - COMPLETE VERSION
elif app_mode == "Manage Data":
    st.markdown('<h2 class="sub-header">🛠️ Manage System Data</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Face Data Management")
        
        if st.session_state.names:
            unique_names = set(st.session_state.names)
            st.write(f"**Registered Individuals:** {len(unique_names)}")
            st.write(f"**Total Face Samples:** {len(st.session_state.names)}")
            if st.session_state.face_data.size > 0:
                st.write(f"**Data Shape:** {st.session_state.face_data.shape}")
            st.write(f"**Color Mode:** {st.session_state.color_mode.upper()}")
            st.write(f"**Features per Face:** {get_expected_features()}")
            st.write(f"**Model Trained:** {'✅ Yes' if st.session_state.model_trained else '❌ No'}")
            
            # Show registered individuals
            st.markdown("#### Registered Individuals:")
            for name in unique_names:
                count = st.session_state.names.count(name)
                st.write(f"- {name}: {count} samples")
        else:
            st.info("No face data registered yet.")
            st.info("Go to 'Add New Face' to register people.")
    
    with col2:
        st.markdown("### 📊 Attendance Data Management")
        
        attendance_files = []
        if os.path.exists('attendance'):
            attendance_files = [f for f in os.listdir('attendance') if f.endswith('.csv')]
        
        st.write(f"**Attendance Files:** {len(attendance_files)}")
        
        if attendance_files:
            total_records = 0
            for file in attendance_files:
                try:
                    file_path = os.path.join('attendance', file)
                    df = pd.read_csv(file_path)
                    total_records += len(df)
                except:
                    pass
            
            st.write(f"**Total Records:** {total_records}")
            
            # Show recent files
            st.markdown("#### Recent Files:")
            for file in attendance_files[:5]:  # Show last 5 files
                st.write(f"- {file}")
        else:
            st.info("No attendance records yet.")
    
    # Data management actions
    st.markdown("### 🛠️ Data Management Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Backup Data")
        if st.button("Export Face Data Backup", use_container_width=True):
            try:
                # Create a zip file with both data files
                import zipfile
                
                with zipfile.ZipFile('face_data_backup.zip', 'w') as zipf:
                    if os.path.exists('data/faces_data.pkl'):
                        zipf.write('data/faces_data.pkl', 'faces_data.pkl')
                    if os.path.exists('data/names.pkl'):
                        zipf.write('data/names.pkl', 'names.pkl')
                    if os.path.exists('data/color_mode.pkl'):
                        zipf.write('data/color_mode.pkl', 'color_mode.pkl')
                
                with open('face_data_backup.zip', 'rb') as f:
                    data = f.read()
                
                st.download_button(
                    label="Download Backup ZIP",
                    data=data,
                    file_name="face_data_backup.zip",
                    mime="application/zip",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error creating backup: {e}")
    
    with col2:
        st.markdown("#### Export Data")
        if st.button("Export All Attendance", use_container_width=True):
            try:
                # Combine all attendance files
                all_attendance = []
                if os.path.exists('attendance'):
                    for file in os.listdir('attendance'):
                        if file.endswith('.csv'):
                            file_path = os.path.join('attendance', file)
                            df = pd.read_csv(file_path)
                            all_attendance.append(df)
                
                if all_attendance:
                    combined_df = pd.concat(all_attendance, ignore_index=True)
                    csv_data = combined_df.to_csv(index=False)
                    
                    st.download_button(
                        label="Download All Attendance CSV",
                        data=csv_data,
                        file_name="all_attendance_records.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.warning("No attendance data to export.")
            except Exception as e:
                st.error(f"Error exporting attendance: {e}")
    
    with col3:
        st.markdown("#### System Maintenance")
        if st.button("Clear All Data", use_container_width=True, type="secondary"):
            st.warning("🚨 DANGER ZONE: This will delete ALL data including faces and attendance!")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.checkbox("I understand this action cannot be undone"):
                    if st.checkbox("I am absolutely sure I want to delete everything"):
                        try:
                            # Clear session state
                            st.session_state.face_data = np.array([])
                            st.session_state.names = []
                            st.session_state.attendance_data = []
                            st.session_state.knn_model = None
                            st.session_state.model_trained = False
                            
                            # Delete data files
                            data_files = ['faces_data.pkl', 'names.pkl', 'color_mode.pkl']
                            for file in data_files:
                                file_path = os.path.join('data', file)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                            
                            # Delete attendance files
                            if os.path.exists('attendance'):
                                for file in os.listdir('attendance'):
                                    file_path = os.path.join('attendance', file)
                                    os.remove(file_path)
                            
                            st.success("✅ All data has been cleared successfully!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error clearing data: {e}")
            
            with col2:
                if st.button("Cancel Clear Operation", use_container_width=True):
                    st.info("Clear operation cancelled.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Face Recognition Attendance System • Built with Streamlit</div>", 
    unsafe_allow_html=True
)