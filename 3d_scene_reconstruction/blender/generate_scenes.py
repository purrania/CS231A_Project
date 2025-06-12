import bpy
import os
import json
import random
import mathutils
from bpy_extras.object_utils import world_to_camera_view
import numpy as np

print("Current Blender file:", bpy.data.filepath)

MODEL_PATHS = {
    'monkey': None,  
    'armadillo': os.path.join(os.path.dirname(__file__), "../assets/Armadillo.blend"),
    'bunny': os.path.join(os.path.dirname(__file__), "../assets/Bunny.blend"),
}

OBJECT_NAMES = {
    'armadillo': "armadillo",  
    'bunny': "bunny",
}

CLASS_MAPPING = {'monkey': 0, 'armadillo': 1, 'bunny': 2}

output_dir = os.path.join(os.path.dirname(__file__), "../data/raw")
image_dir = os.path.join(output_dir, "images")
annotation_dir = os.path.join(output_dir, "annotations")
label_dir = os.path.join(output_dir, "labels")
os.makedirs(image_dir, exist_ok=True)
os.makedirs(annotation_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

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
    coords = [mat_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    coords_camera = [world_to_camera_view(scene, cam, co) for co in coords]
    coords_camera = [co for co in coords_camera if 0.0 <= co.x <= 1.0 and 0.0 <= co.y <= 1.0 and co.z >= 0.0]
    if not coords_camera:
        return None
    xs = [co.x for co in coords_camera]
    ys = [co.y for co in coords_camera]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    x_center = (xmin + xmax) / 2
    y_center = (ymin + ymax) / 2
    width = xmax - xmin
    height = ymax - ymin
    return [x_center, 1 - y_center, width, height]

def is_object_fully_in_frame(obj, cam, scene):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mat_world = obj_eval.matrix_world
    coords = [mat_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    for co in coords:
        co_cam = world_to_camera_view(scene, cam, co)
        if co_cam.x < 0.0 or co_cam.x > 1.0 or co_cam.y < 0.0 or co_cam.y > 1.0 or co_cam.z < 0.0:
            return False
    return True

def append_object_from_blend(blend_filepath, object_name):
    print(f"Appending '{object_name}' from {blend_filepath}")
    existing_objs = set(bpy.data.objects.keys())
    bpy.ops.wm.append(
        filepath=os.path.join(blend_filepath, "Object", object_name),
        directory=os.path.join(blend_filepath, "Object"),
        filename=object_name,
    )
    new_objs = set(bpy.data.objects.keys()) - existing_objs
    if not new_objs:
        raise RuntimeError(f"Failed to append object '{object_name}' from {blend_filepath}")
    new_obj_name = new_objs.pop()
    print(f"Appended new object: {new_obj_name}")
    return bpy.data.objects[new_obj_name]

def ensure_basic_material(obj, color=(0.8, 0.2, 0.2, 1)):
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="DefaultMaterial")
        mat.diffuse_color = color
        obj.data.materials.append(mat)

def add_object(obj_type):
    if obj_type == 'monkey':
        bpy.ops.mesh.primitive_monkey_add()
        obj = bpy.context.object
    else:
        blend_path = MODEL_PATHS[obj_type]
        object_name = OBJECT_NAMES[obj_type]
        obj = append_object_from_blend(blend_path, object_name)
    ensure_basic_material(obj)
    return obj

def generate_scene(scene_idx):
    clear_scene()
    setup_light()
    cam = setup_camera()
    bpy.context.scene.render.resolution_x = 640
    bpy.context.scene.render.resolution_y = 480
    bpy.context.scene.render.resolution_percentage = 100

    obj_type = random.choice(list(MODEL_PATHS.keys()))
    objects, classes, bboxes, orientations = [], [], [], []

    max_attempts = 15
    obj = None

    for attempt in range(max_attempts):
        if obj:
            bpy.data.objects.remove(obj, do_unlink=True)
        obj = add_object(obj_type)
        if not obj or len(obj.data.vertices) == 0:
            print(f"[WARN] Skipping empty object: {obj_type}")
            continue

        obj.scale = (0.5, 0.5, 0.5)
        rot = (
            random.uniform(0, 3.14),
            random.uniform(0, 3.14),
            random.uniform(0, 3.14)
        )
        obj.rotation_euler = rot
        obj.location = (
            random.uniform(-0.3, 0.3),
            random.uniform(-0.3, 0.3),
            random.uniform(0.5, 1.5)
        )

        if is_object_fully_in_frame(obj, cam, bpy.context.scene):
            bbox = get_2d_bbox_yolo(obj, cam, bpy.context.scene)
            if bbox:
                objects.append(obj)
                classes.append(obj_type)
                bboxes.append(bbox)
                orientations.append(rot)
                break

    if not objects and obj:
        bbox = get_2d_bbox_yolo(obj, cam, bpy.context.scene)
        if bbox:
            objects.append(obj)
            classes.append(obj_type)
            bboxes.append(bbox)
            orientations.append(obj.rotation_euler)

    image_filename = f"scene_{scene_idx:04d}.png"
    image_path = os.path.join(image_dir, image_filename)
    bpy.context.scene.render.filepath = image_path
    bpy.ops.render.render(write_still=True)

    annotation = {
        "image_path": image_filename,
        "bboxes": bboxes,
        "classes": classes,
        "orientations": [list(rot) for rot in orientations]
    }

    with open(os.path.join(annotation_dir, f"scene_{scene_idx:04d}.json"), "w") as f:
        json.dump(annotation, f, indent=2)

    with open(os.path.join(label_dir, f"scene_{scene_idx:04d}.txt"), "w") as f:
        for cls_name, bbox in zip(classes, bboxes):
            cls_id = CLASS_MAPPING[cls_name]
            bbox_str = " ".join(f"{x:.6f}" for x in bbox)
            f.write(f"{cls_id} {bbox_str}\n")

    print(f"Rendered {image_filename} with {len(objects)} objects, saved annotation and labels.")


