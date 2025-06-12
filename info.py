import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import tempfile

# Load YOLO model
model = YOLO("best.pt")

# Streamlit app title
st.title("BANI YOLOv8 Object Detection")

# Sidebar configuration
st.sidebar.header("Model Configuration")
selected_task = st.sidebar.radio("Select Task", ("Detection", "Segmentation"), index=0)
confidence = st.sidebar.slider("Model Confidence", min_value=25, max_value=100, value=40)

st.sidebar.header("Input Source")
source = st.sidebar.radio("Select Source", ("Image", "Video", "Webcam"), index=0)


object_info = {
    "1st aid kit": "A container holding medical supplies for emergency treatments.",
    "Aircon": "An appliance that cools and regulates indoor air temperature.",
    "Arduino_uno": "A microcontroller board used for prototyping and electronics projects.",
    "Breadboard": "A board for prototyping circuits without soldering.",
    "Button": "A small switch that closes or opens a circuit when pressed.",
    "Capacitor": "An electronic component that stores and releases electrical energy",
    "Ceiling Fan": "A fan mounted on the ceiling for air circulation.",
    "Chair": "A piece of furniture designed for sitting.",
    "Chassis": "The structural frame for housing components in electronics or vehicles.",
    "Crimping tool": "A tool for attaching connectors to cables or wires.",
    "DC Motor": "A motor that runs on direct current electricity.",
    "IC": "Integrated Circuit; a small chip containing electronic circuits.",
    "ID": "Identification card used for personal identification.",
    "Inductor": "An electronic component that stores energy in a magnetic field.",
    "Jumper Wires": "Wires used to connect components in a circuit.",
    "Keypad": "An input device with multiple buttons for data entry.",
    "LCD": "Liquid Crystal Display; used for visual output in devices.",
    "Led": "Light Emitting Diode; a semiconductor light source.",
    "Multimeter": "A tool for measuring electrical properties like voltage, current, and resistance.",
    "NI-Elvis": "National Instruments Educational Laboratory Virtual Instrumentation Suite; used for engineering experiments.",
    "PCB": "Printed Circuit Board; holds and connects electronic components.",
    "Power supply": "A device providing electrical power to circuits or devices.",
    "Raspberry-Pi-4": "A small computer used for development and educational projects.",
    "Raspberry-Pi-5": "The fifth-generation Raspberry Pi with enhanced performance and features.",
    "Servomotor": "A motor with precise control for position and speed.",
    "Seven-segment": "A display device with seven LEDs used to show numbers.",
    "Table": "A piece of furniture with a flat surface for working or placing items.",
    "Television": "An electronic device for viewing broadcasted or streamed media.",
    "Transistor": "A semiconductor device for amplifying or switching electronic signals.",
    "Tumbler": "A reusable container for holding liquids, often insulated.",
    "Ultrasonic Sensor": "A sensor that measures distance using ultrasonic sound waves.",
    "door": "A hinged or sliding barrier for entry or exit.",
    "fan switch": "A switch for controlling a fan's operation.",
    "keyboard": "An input device for typing and command entry.",
    "lanyard": "A cord worn around the neck to hold an ID or keys.",
    "laptop": "A portable personal computer with an integrated screen and keyboard.",
    "light switch": "A switch for turning lights on or off.",
    "mobile phone": "A portable device for communication and multimedia use.",
    "monitor": "A screen for displaying computer or video output.",
    "mouse": "A pointing device used to interact with a computer interface.",
    "necktie": "A piece of fabric worn around the neck, usually for formal attire.",
    "person": "A human being regarded as an individual.",
    "photoresistor": "A light-sensitive resistor that changes resistance based on light intensity.",
    "potentiometer": "A variable resistor used for adjusting voltage in circuits.",
    "resistor": "An electronic component that limits electrical current in a circuit.",
    "tarpaulin": "A large sheet of waterproof material often used for outdoor advertising or covering items.",
    "trash bin": "A container for collecting garbage.",
    "whiteboard": "A glossy board for writing with erasable markers."
}


# Initialize session state
if "unique_objects" not in st.session_state:
    st.session_state["unique_objects"] = {}  # Store detected objects with confidence
if "selected_object" not in st.session_state:
    st.session_state["selected_object"] = None  # Store the currently selected object

# Reset session state for new source
def reset_session_state():
    st.session_state["unique_objects"] = {}
    st.session_state["selected_object"] = None

# Reset state when switching input sources
reset_session_state()

# Process a frame and update session state
def process_frame(frame, frame_source="frame"):
    """Process a frame (image or video) for object detection."""
    results = model.predict(source=frame, conf=confidence / 100.0, save=False)
    draw_image = frame.copy() if frame_source == "image" else frame
    draw = ImageDraw.Draw(draw_image) if frame_source == "image" else None

    for box in results[0].boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item()) * 100  # Convert confidence to percentage
        label = results[0].names[cls]

        # Add object if not already in session state
        if label not in st.session_state["unique_objects"]:
            st.session_state["unique_objects"][label] = conf

        # Draw bounding boxes
        if frame_source == "image":
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
            draw.text((x1, y1 - 10), f"{label} ({conf:.2f}%)", fill="red")
        elif frame_source == "frame":
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return draw_image if frame_source == "image" else frame

# Display object buttons
def display_object_buttons(frame_id):
    """Display buttons for detected objects."""
    unique_objects = st.session_state["unique_objects"]
    if unique_objects:
        columns = st.columns(len(unique_objects))  # Create columns for detected objects
        for idx, (label, conf) in enumerate(unique_objects.items()):
            with columns[idx]:
                # Ensure each button key is unique by appending frame_id
                button_key = f"button_{label}_{conf:.2f}_{frame_id}"
                if st.button(f"{label} ({conf:.2f}%)", key=button_key):
                    st.session_state["selected_object"] = label  # Update selected object

# Display selected object information
def display_selected_object_info():
    """Display information about the selected object."""
    selected_object = st.session_state["selected_object"]
    if selected_object:
        st.subheader(f"Information about {selected_object}")
        st.write(object_info.get(selected_object, "No information available for this object."))

# Handle input sources
if source == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if selected_task == "Detection":
            processed_image = process_frame(image, frame_source="image")
            st.image(processed_image, caption="Detected Objects", use_container_width=True)
            display_object_buttons()
            display_selected_object_info()

elif source == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        st.video(temp_video.name)

        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()
        reset_session_state()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = process_frame(frame, frame_source="frame")
            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            display_object_buttons()
            display_selected_object_info()

        cap.release()

# Handling live webcam input
elif source == "Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    reset_session_state()  # Reset detected objects for new webcam session
    frame_id = 0  # Initialize frame counter

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = process_frame(frame, frame_source="frame")
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        display_object_buttons(frame_id)  # Pass frame_id to ensure unique keys
        display_selected_object_info()
        
        frame_id += 1  # Increment frame ID for each frame

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()