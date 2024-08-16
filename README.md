# Bear-Detection

## Overview
This repository contains code for detecting and classifying bears in images and videos using deep learning models. The project includes two main parts:

- Bear Classification Model: A TensorFlow/Keras model to identify camera trap images where bears are present
- Bear Detection Model: A YOLOv8 model for detecting bears in trail cam videos.

## Table of Contents
Requirements
Dataset
Bear Classification Model
Bear Detection Model
Video Processing
Setup and Usage
License

To set up the project, you'll need to install several dependencies. You can do this using pip. Here’s a list of required packages:

```
pip install tensorflow keras opencv-python
pip install tensorflow-addons
pip install --upgrade typeguard
pip install ultralytics==8.0.196
pip install roboflow
pip install pyyaml
```
## Dataset
The dataset used in this project is designed for bear detection. It consists of images classified into "bear" and "other" categories and is used for both training and validation.

### Data Preparation
#### - Clone the Repository:

```
git clone https://github.com/danort92/Bear-Detection.git
cd Bear-Detection
```
#### - Data Augmentation and Splitting:
The dataset is split into training and validation sets. Images are augmented to improve model performance.
```
# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)

```
## Binary Classification Model
### Overview
A MobileNetV2-based model is used for binary classification ("Bear" vs. "Other"). The model is trained with early stopping and class weights.

#### Model Training:
##### - Model Training
```
K.clear_session()

# Compute class weights
class_weights = class_weight.compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))

# Load pre-trained model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Recall()])

# Define the early stopping callback
early_stopping = EarlyStopping(
    monitor='val_accuracy',    # Monitor validation accuracy
    patience=2,
    restore_best_weights=True  # Restore the best weights
)

# Train the model with early stopping
history = model.fit(
    train_generator,
    epochs=3,
    validation_data=val_generator,
    callbacks=[early_stopping],
    class_weight=class_weights
)
```
##### - Model Evaluation:
Evaluate the model with custom thresholds and visualize the results.
```
def evaluate_with_threshold(model, generator, threshold=0.5):
    """
    Evaluate the model using a custom threshold and compute metrics.

    Args:
        model: Trained Keras model.
        generator: Data generator to provide images and labels.
        threshold: Classification threshold.
    """
    # Predict labels for the entire validation set
    all_predictions = []
    all_true_labels = []

    # Ensure generator starts from the beginning
    generator.reset()

    # Iterate over the generator to get all images and labels
    for _ in range(len(generator)):
        images, true_labels = next(generator)
        predictions = model.predict(images)
        all_predictions.extend(predictions.flatten())
        all_true_labels.extend(true_labels.flatten())

    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)

    # Apply threshold to predictions
    predicted_labels = (all_predictions > threshold).astype(int)

    # Compute confusion matrix
    cm = confusion_matrix(all_true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Other', 'Bear'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix (Threshold = {threshold})')
    plt.show()

    # Print classification report
    report = classification_report(all_true_labels, predicted_labels, target_names=['Other', 'Bear'])
    print(f"\n Classification Report (Threshold = {threshold}):\n{report}")

    return predicted_labels, all_true_labels

predicted_labels, true_labels = evaluate_with_threshold(model, val_generator, threshold=0.3)
```

## Bear Detection Model
### - Setup and Train YOLOv8 Model
YOLOv8 is used for detecting bears in images and videos. The dataset is downloaded from Roboflow and configured with a dynamically generated data.yaml file.
You are asked your Roboflow credentials to upload your own labelled dataset to train the model.
```
# Function to find the data.yaml file in a given directory
def find_yaml_file(directory, filename="data.yaml"):
    for root, dirs, files in os.walk(directory):
        if filename in files:
            return os.path.join(root, filename)
    return None

# Function to find the train and validation folders automatically
def find_data_folders(base_dir):
    train_dir = None
    val_dir = None

    for dirpath, dirnames, filenames in os.walk(base_dir):
        if "train" in dirpath and "images" in dirpath:
            train_dir = dirpath
        elif "test" in dirpath and "images" in dirpath:
            val_dir = dirpath

    return train_dir, val_dir

def setup_bear_detection(api_key, workspace_name, project_name, version_number):
    # Initialize Roboflow with the provided API key
    rf = Roboflow(api_key=api_key)

    # Access the project in the specified workspace
    project = rf.workspace(workspace_name).project(project_name)

    # Download the specified version of the dataset
    version = project.version(version_number)
    dataset = version.download("yolov8")

    # Define the base directory where the dataset was extracted
    base_dir = f"/content/Bear-Detection/Bear-detection-{version_number}/"

    # Automatically find the data.yaml file in the extracted dataset folder
    yaml_file_path = find_yaml_file(base_dir)

    # Automatically find the train and validation paths
    train_path, val_path = find_data_folders(base_dir)

    if yaml_file_path and train_path and val_path:
        # Load the existing data.yaml file
        with open(yaml_file_path, 'r') as file:
            data = yaml.safe_load(file)

        # Modify the paths to the train and validation datasets
        data['train'] = train_path
        data['val'] = val_path

        # Save the modified data.yaml file
        with open(yaml_file_path, 'w') as file:
            yaml.safe_dump(data, file)

        # Return the path to the data.yaml file and the version number for further use
        return yaml_file_path, version_number

    else:
        raise FileNotFoundError("Required files or directories not found.")

# Ask the user for the Roboflow details once
api_key = input("Enter your Roboflow API key: ")
workspace_name = input("Enter your Roboflow workspace name: ")
project_name = input("Enter your Roboflow project name: ")
version_number = int(input("Enter the version number of the dataset: "))

# Run the setup function and get the data.yaml path and version number
data_yaml_path, version_number = setup_bear_detection(api_key, workspace_name, project_name, version_number)

# Load YOLOv8 model
loaded_model = YOLO("yolov8n.pt") 
```
### - Model Training:
```
# Train the model using the obtained data.yaml path
loaded_model.train(
    data=data_yaml_path,  
    epochs=5,
    imgsz=416, 
    batch=16, 
    optimizer="AdamW", 
    lr0=0.001, 
    weight_decay=0.0005, 
    augment=False, 
    half=True 
)
```
### - Processing Videos:

The function process_video_with_yolo processes video files and adds bounding boxes to detected bears.
# Function to process video with YOLO
```
def process_video_with_yolo(video_path, model, output_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a VideoWriter object if output_path is provided
    if output_path:
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # Suppress output from the model prediction
    logging.getLogger('ultralytics').setLevel(logging.ERROR)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model.predict(frame_rgb, verbose=False)  # Suppress verbose output

        # Draw bounding boxes
        for bbox in results[0].boxes.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, 'Predicted BB', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv2.LINE_AA)

        # Write the frame to the output video if output_path is provided
        if output_path:
            out.write(frame)

    cap.release()
    if output_path:
        out.release()
    cv2.destroyAllWindows()

def setup_and_process_videos():
    # Directories
    video_files_dir = '/content/Bear-Detection/video_files'
    processed_videos_dir = '/content/Bear-Detection/processed_videos'

    # Create directories if they do not exist
    os.makedirs(video_files_dir, exist_ok=True)
    os.makedirs(processed_videos_dir, exist_ok=True)

    # Upload video files
    print("Upload video files:")
    uploaded = files.upload()

    # List uploaded files
    video_files = list(uploaded.keys())
    print("Uploaded video files:", video_files)

    # Save uploaded files to video_files_dir
    for video_file in video_files:
        file_path = os.path.join(video_files_dir, video_file)
        with open(file_path, 'wb') as f:
            f.write(uploaded[video_file])
        print(f"Uploaded and saved {video_file} to {video_files_dir}")

    # Automatically find all .mp4 files in the video_files_dir
    video_files = [os.path.join(video_files_dir, f) for f in os.listdir(video_files_dir) if f.endswith('.mp4')]

    # Load your model here
    model = YOLO("yolov8n.pt")  # Adjust model as needed

    # Process each video file and save the output
    for video_file in video_files:
        output_file = os.path.join(processed_videos_dir, os.path.basename(video_file).replace('.mp4', '_processed.mp4'))
        print(f"Processing video: {video_file}")
        process_video_with_yolo(video_file, model, output_file)
        print(f"Processed video saved as: {output_file}")

    # Print the directory where the processed videos are saved
    print(f"\nAll processed videos are saved in: {processed_videos_dir}")

# Run the setup and processing function
setup_and_process_videos()
```
## Setup and Usage
### Setup
#### - Clone the Repository:

```
git clone https://github.com/yourusername/Bear-Detection.git
cd Bear-Detection
```
#### - Install Dependencies:
Follow the installation instructions in the Requirements section.

#### - Download Dataset:
Follow the instructions in the Dataset section.

## Usage
### Train the Classification Model:
Run the provided scripts to train and evaluate the classification model.

### Train the Detection Model:
Follow the instructions to set up and train the YOLOv8 model.

### Process Videos:
Upload videos and use the provided function to process and analyze them.

Examples:

![ezgif-7-c8751c7a2e](https://github.com/user-attachments/assets/ea41e5cd-641e-4a36-87c7-b3cdf565cd6e)

![ezgif-7-068f08d35c (1) (1) (2)](https://github.com/user-attachments/assets/9c947f16-ff53-4aa6-b254-58b9583aade8)

## License
This project is licensed under the MIT License. See the LICENSE file for details.

