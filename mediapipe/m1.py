# Load image
import mediapipe as mp
import cv2
image = cv2.imread('ram.jpg')

# Create a face mesh object
mp_face_mesh = mp.solutions.face_mesh

# Run face mesh on the image
with mp_face_mesh.FaceMesh(static_image_mode=False,max_num_faces=2,refine_landmarks=True,min_detection_confidence=0.5) as face_mesh:
    results=face_mesh.process(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))

# Get the mesh coordinates
mesh_coords = results.multi_face_landmarks[0]

# Get the triangles
triangles = mp_face_mesh.FACEMESH_TESSELATION

# Iterate over all triangles
for i, triangle in enumerate(triangles):
    print(f'Triangle {i}: {triangle}')