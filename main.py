import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Initialize the face mesh model
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

# Read an image from file
image = cv2.imread('ram.jpg')

# Convert the image to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect the face mesh
results = face_mesh.process(image)

# Draw the face mesh on the image with node numbering
if results.multi_face_landmarks:
    for face_landmarks in results.multi_face_landmarks:
        # Draw the face mesh with connections
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),
        )
        # Add node numbering to the face mesh
        for i, landmark in enumerate(face_landmarks.landmark):
            x, y = int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])
            cv2.putText(image, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

# Display the image
cv2.imshow('Face Mesh with Node Numbering', image)
cv2.waitKey(0)

# Release resources
face_mesh.close()
cv2.destroyAllWindows()
