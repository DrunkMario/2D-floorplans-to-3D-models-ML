import bpy
import bmesh

# Function to create a wall from contour points
def create_wall_from_contour(contour, height=2.5):
    # Create a new mesh and object
    mesh = bpy.data.meshes.new("Wall")
    obj = bpy.data.objects.new("Wall", mesh)

    # Link object to the scene
    bpy.context.collection.objects.link(obj)

    # Create a bmesh to build geometry
    bm = bmesh.new()

    # Create a list of vertices
    verts = []
    for point in contour:
        # Create bottom vertices
        v1 = bm.verts.new((point[0], point[1], 0))
        v2 = bm.verts.new((point[0], point[1], height))
        verts.append((v1, v2))

    # Create faces for the walls
    for i in range(len(verts) - 1):
        # Create side faces
        bm.faces.new((verts[i][0], verts[i + 1][0], verts[i + 1][1], verts[i][1]))
    
    # Finalize the bmesh
    bmesh.ops.recalc_face_normals(bm)
    bm.to_mesh(mesh)
    bm.free()

# Read contours from file and create walls
with open('contours.txt', 'r') as f:
    for line in f:
        # Parse contour points
        points = [tuple(map(float, point.split(','))) for point in line.strip().split(',')]
        create_wall_from_contour(points)

# Optionally select all walls in the scene
bpy.ops.object.select_all(action='DESELECT')
for obj in bpy.context.collection.objects:
    if "Wall" in obj.name:
        obj.select_set(True)
