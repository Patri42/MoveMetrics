import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Choose the model and set input size
model_name = "movenet_lightning"
input_size = 192 if "lightning" in model_name else 256

# Load the model from TensorFlow Hub
if "movenet_lightning" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
elif "movenet_thunder" in model_name:
    module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
else:
    raise ValueError("Unsupported model name: %s" % model_name)

def movenet(input_image):
    model = module.signatures['serving_default']
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model(input_image)
    return outputs['output_0'].numpy()

# Dictionary for mapping keypoint indices to keypoints names
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# Dictionary for mapping bones to a matplotlib color name
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): (255, 0, 255), (0, 2): (0, 255, 255), (1, 3): (255, 0, 255), (2, 4): (0, 255, 255),
    (0, 5): (255, 0, 255), (0, 6): (0, 255, 255), (5, 7): (255, 0, 255), (7, 9): (255, 0, 255),
    (6, 8): (0, 255, 255), (8, 10): (0, 255, 255), (5, 6): (255, 255, 0), (5, 11): (255, 0, 255),
    (6, 12): (0, 255, 255), (11, 12): (255, 255, 0), (11, 13): (255, 0, 255), (13, 15): (255, 0, 255),
    (12, 14): (0, 255, 255), (14, 16): (0, 255, 255)
}

def draw_keypoints_and_edges(frame, keypoints, edges, keypoint_threshold=0.3):
    y, x, _ = frame.shape
    shaped = np.squeeze(keypoints)

    # Draw keypoints
    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > keypoint_threshold:
            cv2.circle(frame, (int(kx * x), int(ky * y)), 5, (0, 255, 0), -1)

    # Draw edges
    for edge, color in edges.items():
        if (shaped[edge[0], 2] > keypoint_threshold and
                shaped[edge[1], 2] > keypoint_threshold):
            pt1 = (int(shaped[edge[0], 1] * x), int(shaped[edge[0], 0] * y))
            pt2 = (int(shaped[edge[1], 1] * x), int(shaped[edge[1], 0] * y))
            cv2.line(frame, pt1, pt2, color, 2)

# Capture video from webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    input_tensor = tf.image.convert_image_dtype(img, dtype=tf.int32)

    # Run pose estimation
    keypoints_with_scores = movenet(input_tensor)

    # Draw keypoints and edges on the frame
    draw_keypoints_and_edges(frame, keypoints_with_scores, KEYPOINT_EDGE_INDS_TO_COLOR)

    cv2.imshow('MoveNet Pose Estimation', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
