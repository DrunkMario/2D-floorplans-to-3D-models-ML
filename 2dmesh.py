import bpy


vertices_2d = [(67, 135), (69, 372), (570, 372), (575, 278), (695, 277), (693, 168), (573, 167), (572, 39), (167, 40), (165, 130)]


vertices_3d = [(x, y, 0.0) for x, y in vertices_2d]

edges = [(i, (i + 1) % len(vertices_2d)) for i in range(len(vertices_2d))]
faces = [[i for i in range(len(vertices_2d))]]  # Single face connecting all vertices


mesh = bpy.data.meshes.new(name="PolygonRoomMesh")
obj = bpy.data.objects.new("PolygonRoom", mesh)

bpy.context.collection.objects.link(obj)

bpy.ops.object.mode_set(mode='OBJECT')

mesh.from_pydata(vertices_3d, edges, faces)
mesh.update()

bpy.ops.object.mode_set(mode='EDIT')

bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value":(0, 0, 3.0)})  # 3 meters height

bpy.ops.object.mode_set(mode='OBJECT')
