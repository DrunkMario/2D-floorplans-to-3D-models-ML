import bpy

# List of tuples containing the (x, y) coordinates of the polygon vertices
vertices_2d = [(67, 135), (69, 372), (570, 372), (575, 278), (695, 277), (693, 168), (573, 167), (572, 39), (167, 40), (165, 130)]

# Add a new mesh and object to the scene
mesh = bpy.data.meshes.new("PolygonMesh")
obj = bpy.data.objects.new("PolygonObject", mesh)

# Link the object to the current collection
bpy.context.collection.objects.link(obj)

# Create the vertices in 3D by converting the 2D coordinates to (x, y, z) with z = 0
vertices_3d = [(x, y, 0) for x, y in vertices_2d]

# Define the face by listing the vertex indices in the order of the vertices
faces = [list(range(len(vertices_3d)))]

# Create the mesh data from the vertices and faces
mesh.from_pydata(vertices_3d, [], faces)

# Update the mesh and recalculate normals
mesh.update(calc_edges=True)

# Optionally, set the object mode to 'EDIT' to check the result
bpy.context.view_layer.objects.active = obj
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.mesh.normals_make_consistent()
bpy.ops.object.mode_set(mode='OBJECT')
