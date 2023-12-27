import cv2
import tensorflow as tf
import numpy as np

KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255, 0, 255),  # Magenta
    (0, 2): (255, 255, 0),  # Cyan
    (1, 3): (255, 0, 255),  # Magenta
    (2, 4): (255, 255, 0),  # Cyan
    (0, 5): (255, 0, 255),  # Magenta
    (0, 6): (255, 255, 0),  # Cyan
    (5, 7): (255, 0, 255),  # Magenta
    (7, 9): (255, 0, 255),  # Magenta
    (6, 8): (255, 255, 0),  # Cyan
    (8, 10): (255, 255, 0), # Cyan
    (5, 6): (0, 255, 255),  # Yellow
    (5, 11): (255, 0, 255), # Magenta
    (6, 12): (255, 255, 0), # Cyan
    (11, 12): (0, 255, 255),# Yellow
    (11, 13): (255, 0, 255),# Magenta
    (13, 15): (255, 0, 255),# Magenta
    (12, 14): (255, 255, 0),# Cyan
    (14, 16): (255, 255, 0) # Cyan
}

def preprocess_frame_for_movenet(frame, input_size=192):
    # Resize the frame to the required input size of the model
    frame_resized = cv2.resize(frame, (input_size, input_size))

    # Convert color space from BGR (OpenCV default) to RGB if needed
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # Normalize pixel values if required by the model
    frame_normalized = frame_rgb / 255.0

    return frame_normalized

def movenet(input_image):
    # TF Lite format expects tensor type of float32.
    input_image = tf.cast(input_image, dtype=tf.float32)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check required input shape and resize if necessary
    input_shape = input_details[0]['shape']
    input_image = tf.image.resize(input_image, (input_shape[1], input_shape[2]))

    # Ensure input_image has shape [1, height, width, 3]
    if len(input_image.shape) == 3:
        input_image = np.expand_dims(input_image, axis=0)

    # Set the tensor to point to the input data to be inferred
    interpreter.set_tensor(input_details[0]['index'], input_image)
    # Run the inference
    interpreter.invoke()

    # Get the model prediction
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def draw_prediction_on_image(image, keypoints_with_scores, keypoint_threshold=0.01):  # Lowered threshold for testing
    height, width, channel = image.shape
    num_instances, _, _, _ = keypoints_with_scores.shape

    for idx in range(num_instances):
        keypoints = keypoints_with_scores[0, idx, :, :]
        keypoints_drawn = 0

        for keypoint in keypoints:
            # Print the confidence score for debugging
            print("Keypoint score:", keypoint[2])

            # Draw the keypoint regardless of the confidence score for testing
            x = int(keypoint[1] * width)
            y = int(keypoint[0] * height)
            cv2.circle(image, (x, y), 5, (0, 255, 0), thickness=-1)

            if keypoint[2] > keypoint_threshold:
                keypoints_drawn += 1

        print(f"Keypoints drawn for instance {idx}: {keypoints_drawn}")

    return image

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Initialize the model (assuming you have already loaded it as in your provided code)
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model Input Details:", input_details)
print("Model Output Details:", output_details)

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame as required by your model
    processed_frame = preprocess_frame_for_movenet(frame)  # Implement this function based on movenet's needs

    # Run the skeleton detection model
    keypoints_with_scores = movenet(processed_frame)
    
    # Draw the skeleton on the frame
    output_frame = draw_prediction_on_image(frame, keypoints_with_scores)

    # Display the resulting frame
    cv2.imshow('Skeleton Detection', output_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
