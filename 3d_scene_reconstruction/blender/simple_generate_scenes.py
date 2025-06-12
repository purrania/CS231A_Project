import bpy
import os
import json
import random
import mathutils
from PIL import Image
from bpy_extras.object_utils import world_to_camera_view
import numpy as np

output_dir = os.path.join(os.path.dirname(__file__), "../data/raw")
image_dir = os.path.join(output_dir, "images")
annotation_dir = os.path.join(output_dir, "annotations")
label_dir = os.path.join(output_dir, "labels")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

OBJECT_TYPES = ['cube', 'uv_sphere', 'cylinder']
CLASS_MAPPING = {'cube': 0, 'uv_sphere': 1, 'cylinder': 2}

def clear_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)

def setup_camera():
    cam_data = bpy.data.cameras.new("Camera")
    cam = bpy.data.objects.new("Camera", cam_data)
    bpy.context.collection.objects.link(cam)
    cam.location = (3, -3, 3)
    cam.rotation_euler = (1.109, 0, 0.814)
    bpy.context.scene.camera = cam
    return cam

def setup_light():
    light_data = bpy.data.lights.new(name="KeyLight", type='AREA')
    light_data.energy = 1000
    light_data.size = 2.0
    light = bpy.data.objects.new(name="KeyLight", object_data=light_data)
    bpy.context.collection.objects.link(light)
    light.location = (5, -5, 5)
    light.rotation_euler = (0.9, 0, 0.8)

    fill_data = bpy.data.lights.new(name="FillLight", type='POINT')
    fill_data.energy = 300
    fill = bpy.data.objects.new(name="FillLight", object_data=fill_data)
    bpy.context.collection.objects.link(fill)
    fill.location = (-4, 4, 2)

def get_2d_bbox_yolo(obj, cam, scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mat_world = obj_eval.matrix_world

    corners_2d = []
    for corner in obj.bound_box:
        world_corner = mat_world @ mathutils.Vector(corner)
        co_ndc = world_to_camera_view(scene, cam, world_corner)
        corners_2d.append(co_ndc)

    xs = [co.x for co in corners_2d]
    ys = [co.y for co in corners_2d]

    xmin = max(min(xs), 0.0)
    xmax = min(max(xs), 1.0)
    ymin = max(min(ys), 0.0)
    ymax = min(max(ys), 1.0)

    if xmin == xmax or ymin == ymax:
        return None

    x_center = (xmin + xmax) / 2
    y_center = 1 - (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin

    return [x_center, y_center, width, height]

def add_object(obj_type):
    if obj_type == 'cube':
        bpy.ops.mesh.primitive_cube_add(size=1)
    elif obj_type == 'uv_sphere':
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.5)
    elif obj_type == 'cylinder':
        bpy.ops.mesh.primitive_cylinder_add(radius=0.5, depth=1)
    else:
        raise ValueError(f"Unknown object type {obj_type}")
    return bpy.context.object

def get_visibility_ratio(obj, scene, width=640, height=480):
    for o in scene.objects:
        if o.type == 'MESH':
            o.hide_render = (o.name != obj.name)

    original_filepath = scene.render.filepath
    tmp_path = "/tmp/vis_mask.png"
    scene.render.filepath = tmp_path
    bpy.ops.render.render(write_still=True)
    scene.render.filepath = original_filepath

    for o in scene.objects:
        o.hide_render = False

    if not os.path.exists(tmp_path):
        return 0.0
    mask_img = Image.open(tmp_path).convert("L")
    mask_arr = np.array(mask_img)
    visible_pixels = np.count_nonzero(mask_arr > 10)
    total_pixels = width * height
    os.remove(tmp_path)
    return visible_pixels / total_pixels

def generate_scene(scene_idx):
    clear_scene()
    setup_light()
    cam = setup_camera()

    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 480
    bpy.context.scene.render.resolution_percentage = 100

    # Change here: 0 or 1 object per scene
    num_objects = random.choice([0, 1])

    objects = []
    classes = []
    bboxes = []

    for _ in range(num_objects):
        obj_type = random.choice(OBJECT_TYPES)
        obj = add_object(obj_type)

        obj.location = (
            random.uniform(-1.5, 1.5),
            random.uniform(-1.5, 1.5),
            random.uniform(0, 1)
        )
        obj.rotation_euler = (
            random.uniform(0, 6.28),
            random.uniform(0, 6.28),
            random.uniform(0, 6.28)
        )
        scale = random.uniform(0.5, 1.5)
        obj.scale = (scale, scale, scale)

        vis_ratio = get_visibility_ratio(obj, bpy.context.scene)
        if vis_ratio < 0.1:
            bpy.data.objects.remove(obj, do_unlink=True)
            continue

        bbox = get_2d_bbox_yolo(obj, cam, bpy.context.scene)
        if bbox:
            objects.append(obj)
            classes.append(obj_type)
            bboxes.append(bbox)
        else:
            bpy.data.objects.remove(obj, do_unlink=True)

    image_filename = f"scene_{scene_idx:04d}.png"
    image_path = os.path.join(image_dir, image_filename)
    bpy.context.scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)

    annotation = {
        "image_path": image_filename,
        "bboxes": bboxes,
        "classes": classes
    }
    annotation_path = os.path.join(annotation_dir, f"scene_{scene_idx:04d}.json")
    with open(annotation_path, "w") as f:
        json.dump(annotation, f, indent=2)

    label_path = os.path.join(label_dir, f"scene_{scene_idx:04d}.txt")
    with open(label_path, "w") as f:
        for cls_name, bbox in zip(classes, bboxes):
            cls_id = CLASS_MAPPING[cls_name]
            bbox_str = " ".join(f"{x:.6f}" for x in bbox)
            f.write(f"{cls_id} {bbox_str}\n")

    print(f"Rendered {image_filename} with {len(objects)} objects, saved annotation and labels.")

def main():
    n_scenes = 100
    print("Starting scene generation...")
    for i in range(n_scenes):
        generate_scene(i)
    print("Scene generation complete!")

if __name__ == "__main__":
    main()

