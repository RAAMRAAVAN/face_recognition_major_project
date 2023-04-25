import mediapipe as mp
import math
import cv2
# Load the Mediapipe face mesh model
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

# Get the face mesh for an input image
image = cv2.imread('ram.jpg')
results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

# Get the mesh vertices and triangle indices
vertices = results.multi_face_landmarks[0].landmark
n_vertices = len(vertices)
triangles = []
for i in range(n_vertices):
    for j in range(i+1, n_vertices):
        for k in range(j+1, n_vertices):
            if (i, j, k) not in triangles:
                # Check if vertices i, j, k form a triangle
                p1, p2, p3 = vertices[i], vertices[j], vertices[k]
                if math.isclose(p1.z, p2.z, rel_tol=1e-3) and math.isclose(p2.z, p3.z, rel_tol=1e-3):
                    triangles.append((i, j, k))

# Calculate the local angle feature for each node
local_angles = []
for i, vertex in enumerate(vertices):
    angles = []
    for triangle in triangles:
        if i in triangle:
            # Calculate the angle at the vertex of the triangle
            p1, p2, p3 = [vertices[i] for i in triangle]
            angle = math.degrees(math.acos((p1.x-p2.x)*(p3.x-p2.x)+(p1.y-p2.y)*(p3.y-p2.y)+(p1.z-p2.z)*(p3.z-p2.z)))
            angles.append(angle)
    if len(angles) > 0:
        local_angles.append(sum(angles) / len(angles))
    else:
        local_angles.append(0.0)
    # local_angles.append(sum(angles) / len(angles))

# Print the local angle feature for each node
for i, angle in enumerate(local_angles):
    print(f"Node {i}: {angle} degrees")
