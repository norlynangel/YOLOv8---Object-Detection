import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import cv2
import tempfile
import requests
import uuid

# Load YOLO model
model = YOLO("best.pt")

# Streamlit app title
st.title("BANI YOLOv8 Object Detection")

# LM Studio API endpoint
LM_STUDIO_URL = "http://localhost:1234/v1/completions"  # Update based on LM Studio configuration


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
    st.session_state["unique_objects"] = {}
if "selected_object" not in st.session_state:
    st.session_state["selected_object"] = None

# Reset session state for new sources
def reset_session_state():
    st.session_state["unique_objects"] = {}
    st.session_state["selected_object"] = None

# Query LM Studio for object details
def query_lm_studio(prompt):
    payload = {
        "model": "llama-3.2-1b-instruct",  # Replace with your model's identifier
        "prompt": prompt,
        "max_tokens": 150,
        "temperature": 0.7
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=60)
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("text", "No response from the model.")
        else:
            error_message = response.json().get("error", "Unknown error")
            return f"LM Studio Error: {error_message}"
    except requests.exceptions.RequestException as e:
        return f"Connection Error: {str(e)}"


# Process a single frame (image or video)
def process_frame(frame, frame_id, frame_source="image"):
    results = model.predict(source=frame, conf=confidence / 100.0, save=False)
    draw_image = frame.copy() if frame_source == "image" else frame

    if frame_source == "image":
        draw = ImageDraw.Draw(draw_image)
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = results[0].names[int(box.cls[0].item())]
            conf = float(box.conf[0].item()) * 100

            key = f"{label}_{conf:.2f}_{frame_id}_{idx}"
            st.session_state["unique_objects"][key] = (label, conf)

            draw.rectangle([x1, y1, x2, y2], outline="DeepPink", width=3)
            draw.text((x1, y1 - 10), f"{label} ({conf:.2f}%)", fill="DeepPink")

    elif frame_source == "video":
        for idx, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = results[0].names[int(box.cls[0].item())]
            conf = float(box.conf[0].item()) * 100

            key = f"{label}_{conf:.2f}_{frame_id}_{idx}"
            st.session_state["unique_objects"][key] = (label, conf)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ({conf:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return draw_image if frame_source == "image" else frame

# Display detected objects and buttons
def display_detected_objects():
    unique_objects = st.session_state["unique_objects"]
    if unique_objects:
        st.sidebar.subheader("Detected Objects")
        for key, (label, conf) in unique_objects.items():
            if st.sidebar.button(f"{label} ({conf:.2f}%)", key=f"button_{key}"):
                st.session_state["selected_object"] = label  # Update selected object information


# Display selected object information
def display_selected_object_info():
    selected_object = st.session_state["selected_object"]
    if selected_object:
        st.subheader(f"Information about {selected_object}")
        info = object_info.get(selected_object, "No local information available.")
        st.write(info)

        # Query LM Studio for additional information
        st.write("Querying LM Studio for more details...")
        lm_response = query_lm_studio(f"Describe the object: {selected_object}")
        st.write(lm_response)


# Add a Chatbot Section in the Sidebar
def chatbot_interface():
    st.sidebar.header("Chatbot")
    user_input = st.sidebar.text_input("Ask about any object or topic:", key="chat_input")  # Persistent input box
    if st.sidebar.button("Submit Query", key="chat_submit"):
        if user_input.strip():
            with st.spinner("Querying..."):
                response = query_lm_studio(user_input)  # Query LM Studio API
                st.session_state["chatbot_response"] = response  # Store the chatbot response

# Initialize chatbot response in session state
if "chatbot_response" not in st.session_state:
    st.session_state["chatbot_response"] = "Ask something using the chatbot!"

# Display Chatbot Response in Main Content
def display_chatbot_response():
    st.subheader("Chatbot Response")
    st.write(st.session_state["chatbot_response"])

# Integrate Chatbot (Input in Sidebar, Response in Main Content)
chatbot_interface()
display_chatbot_response()  # Display the chatbot's response in the main content



# Handle input sources
if source == "Image":
    uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        processed_image = process_frame(image, frame_id="static", frame_source="image")
        st.image(processed_image, caption="Detected Objects", use_container_width=True)
        display_detected_objects()
        display_selected_object_info()

elif source == "Video":
    uploaded_video = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        temp_video = tempfile.NamedTemporaryFile(delete=False)
        temp_video.write(uploaded_video.read())
        st.video(temp_video.name)

        cap = cv2.VideoCapture(temp_video.name)
        stframe = st.empty()
        frame_id = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            processed_frame = process_frame(frame, frame_id=frame_id, frame_source="video")
            stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            display_detected_objects()
            display_selected_object_info()
            frame_id += 1

        cap.release()

elif source == "Webcam":
    cap = cv2.VideoCapture(0)
    stframe = st.empty()
    frame_id = 0

    reset_session_state()  # Reset detected objects for the new webcam session

    # Sidebar initialization
    st.sidebar.subheader("Detected Objects")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame and update session state
        processed_frame = process_frame(frame, frame_id=frame_id, frame_source="video")
        stframe.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Update detected objects in the sidebar
        unique_objects = st.session_state["unique_objects"]
        if unique_objects:
            for key, (label, conf) in unique_objects.items():
                button_key = f"button_{label}_{conf:.2f}_{frame_id}"
                if st.sidebar.button(f"{label} ({conf:.2f}%)", key=button_key):
                    st.session_state["selected_object"] = label

        # Display selected object information
        display_selected_object_info()
        frame_id += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
