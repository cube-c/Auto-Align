import numpy as np
import bpy
bl_info = {
    'name': 'Auto Align',
    "author": 'cubec',
    'blender': (3, 1, 2),
    'category': 'Object',
    "description": "Automatically Align Selected Objects Parallel to World Axis",
}


# Hyperparameters
ITERATION = 100
THRESHOLD = 5 * (np.pi/180)
MAX_POLYS = 10000


class ObjectAutoAlign(bpy.types.Operator):
    bl_idname = 'object.auto_align'
    bl_label = 'Auto Align'
    bl_description = 'Automatically align selected objects parallel to world axis'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        for m in bpy.context.selected_objects:
            if m.type != "MESH":
                continue

            polys = m.data.polygons
            if len(polys) == 0:
                continue

            areas = np.array([p.area for p in polys])
            normals = np.array([list(p.normal) for p in polys])
            model = get_matrix(areas, normals)

            for v in m.data.vertices:
                v.co = model @ v.co

        return {'FINISHED'}


def get_matrix(areas, normals):
    # Resample if too many polygons
    if areas.size > MAX_POLYS:
        indices = np.random.choice(
            areas.size, MAX_POLYS, p=areas/sum(areas), replace=False)
        areas = areas[indices]
        normals = normals[indices]

    indices = np.random.choice(areas.size, ITERATION, p=areas/sum(areas))

    # RANSAC
    best_model = np.identity(3)
    best_value = 0

    for index in indices:
        model = np.zeros((3, 3))
        model[0] = normals[index]
        next_indices = np.nonzero(
            np.abs(normals@model[0]) < np.sin(THRESHOLD))[0]
        if next_indices.size > 0:
            next_areas = areas[next_indices]
            model[1] = normals[np.random.choice(
                next_indices, p=next_areas/sum(next_areas))]
        else:
            model[1] = np.zeros(3)
            model[1][(np.argmax(np.abs(model[0]))+1) % 3] = 1

        model[1] = np.cross(model[0], model[1])
        model[1] = model[1] / np.linalg.norm(model[1])
        model[2] = np.cross(model[0], model[1])

        value = np.sum(
            areas[np.max(np.abs(normals@model.T), axis=1) > np.cos(THRESHOLD)])
        if best_value < value:
            best_value, best_model = value, model

    # Least-square adjustment
    A = np.zeros((3, 3))
    dist = normals@best_model.T
    weighted_normals = np.expand_dims(areas, axis=1) * normals

    for i in range(3):
        selected_indices = (dist[:, i] > np.cos(THRESHOLD))
        A[i] += np.sum(weighted_normals[selected_indices, :], axis=0)
        selected_indices = (dist[:, i] < -np.cos(THRESHOLD))
        A[i] -= np.sum(weighted_normals[selected_indices, :], axis=0)

    if np.linalg.cond(A) < 1/np.finfo(A.dtype).eps:
        evalues, evectors = np.linalg.eigh(A.T@A)
        best_model = A @ (evectors @ np.diag(np.sqrt(1/evalues)) @ evectors.T)

    # Find minimal rotation matrix
    unit_rot = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    flip_rot = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    unit_diag = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]])

    best_model_opt = best_model
    best_trace = 0
    rot = np.identity(3)
    for _ in range(3):
        rot = unit_rot@rot
        for j in range(4):
            model_opt = np.diag(unit_diag[j]) @ rot @ best_model
            trace = np.trace(model_opt)
            if trace > best_trace:
                best_trace, best_model_opt = trace, model_opt

            model_opt = -np.diag(unit_diag[j]) @ flip_rot @ rot @ best_model
            trace = np.trace(model_opt)
            if trace > best_trace:
                best_trace, best_model_opt = trace, model_opt

    return best_model_opt


def menu_func(self, context):
    self.layout.operator(ObjectAutoAlign.bl_idname)


def register():
    bpy.utils.register_class(ObjectAutoAlign)
    bpy.types.VIEW3D_MT_object.append(menu_func)


def unregister():
    bpy.utils.unregister_class(ObjectAutoAlign)
    bpy.types.VIEW3D_MT_object.remove(menu_func)


if __name__ == '__main__':
    register()
