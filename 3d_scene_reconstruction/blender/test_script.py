print("Hello from Blender script!")

import bpy

bpy.ops.mesh.primitive_cube_add(size=2)

bpy.context.scene.render.filepath = "//../data/raw/images/test_render.png"

bpy.ops.render.render(write_still=True)

print("Render complete!")

