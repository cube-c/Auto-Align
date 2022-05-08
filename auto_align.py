import numpy as np
import bpy
bl_info = {
    'name': 'Auto Align',
    "author": 'cubec',
    'blender': (3, 1, 2),
    'version': (0, 4, 0),
    'category': 'Object',
    "description": "Automatically Align Selected Objects Parallel to World Axis",
}

# Hyperparameters
ITERATION_RANSAC = 100
ITERATION_MEDIAN = 10
THRESHOLD = 5 * (np.pi/180)
MAX_POLYS = 10000


class OBJECT_OT_AutoAlignOperator(bpy.types.Operator):
    bl_idname = 'object.auto_align'
    bl_label = 'Auto Align'
    bl_description = 'Align selected objects parallel to world axis'
    bl_options = {'REGISTER', 'UNDO'}

    bake: bpy.props.BoolProperty(default=False, name='Bake')
    keep: bpy.props.BoolProperty(default=False, name='Keep')

    def execute(self, context):
        align(context, keep=self.keep, bake=self.bake)
        return {'FINISHED'}


def align(context, bake=False, keep=False):
    keep_bucket = []
    for m in context.selected_objects:
        if m.type != "MESH":
            continue

        polys = m.data.polygons
        if len(polys) == 0:
            continue

        global_matrix = np.array(m.matrix_basis)
        areas = np.array([p.area for p in polys])
        normals = np.array([list(p.normal)
                            for p in polys]) @ global_matrix[:3, :3].T
        normals = normals / \
            np.expand_dims(np.linalg.norm(normals, axis=1), axis=1)

        model = get_matrix(areas, normals)
        global_matrix[:3, :3] = model@global_matrix[:3, :3]

        m.matrix_basis = global_matrix.T

        if keep:
            keep_bucket.append((m, model))

    if bake:
        bpy.ops.object.transform_apply(
            location=False, rotation=True, scale=False)

    if keep:
        for m, model in keep_bucket:
            global_matrix = np.array(m.matrix_basis)
            global_matrix[:3, :3] = model.T@global_matrix[:3, :3]
            m.matrix_basis = global_matrix.T


def get_matrix(areas, normals):
    # Resample if too many polygons
    if areas.size > MAX_POLYS:
        indices = np.random.choice(
            areas.size, MAX_POLYS, p=areas/sum(areas), replace=False)
        areas = areas[indices]
        normals = normals[indices]

    first_indices = np.random.choice(
        areas.size, ITERATION_RANSAC, p=areas/sum(areas))

    # RANSAC
    best_model = np.identity(3)
    best_value = 0

    for index in first_indices:
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

        indices = np.max(np.abs(normals@model.T), axis=1) > np.cos(THRESHOLD)
        value = np.sum(areas[indices])
        if best_value < value:
            best_value, best_model, best_indices = value, model, indices

    # Calculate median each axis, iteratively...
    areas = areas[best_indices]
    normals = normals[best_indices]
    axis = np.vstack((best_model, -best_model))
    axis_indices = np.argmax(normals@axis.T, axis=1)
    normals_per_axis = []
    areas_per_axis = []
    xyz_axis = np.array([[[1, 2], [2, 4], [4, 5], [5, 1]], [[3, 2], [2, 0], [
                        0, 5], [5, 3]], [[0, 1], [1, 3], [3, 4], [4, 0]]])
    for i in range(6):
        normals_per_axis.append(normals[axis_indices == i])
        areas_per_axis.append(areas[axis_indices == i])

    normals_area = []
    for i in range(3):
        normals_area.append(np.concatenate(
            [areas_per_axis[a] for (a, _) in xyz_axis[i]]))

    for _ in range(ITERATION_MEDIAN):
        for i in range(3):
            normals_proj = np.concatenate(
                [normals_per_axis[a] @ axis[b] for (a, b) in xyz_axis[i]])
            sort_indices = np.argsort(normals_proj)
            value = normals_proj[sort_indices]
            weight = normals_area[i][sort_indices]
            weight_cumsum = np.cumsum(weight)
            med_index = np.searchsorted(weight_cumsum, weight_cumsum[-1]/2)

            c, s = np.cos(value[med_index]), np.sin(value[med_index])
            j, k = (i+1) % 3, (i+2) % 3

            transform = np.identity(3)
            transform[(j, j, k, k), (j, k, j, k)] = np.array([c, -s, s, c])
            best_model = transform.T@best_model
            axis = np.vstack((best_model, -best_model))

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


class VIEW3D_PT_AutoAlignUi(bpy.types.Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_label = 'Auto Align'
    bl_context = 'objectmode'
    bl_category = 'Item'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout

        row0 = layout.row()
        prop0 = row0.operator(
            OBJECT_OT_AutoAlignOperator.bl_idname, text='Rotate')
        prop0.bake, prop0.keep = False, False

        row1 = layout.row()
        prop1 = row1.operator(
            OBJECT_OT_AutoAlignOperator.bl_idname, text='Rotate & Bake')
        prop1.bake, prop1.keep = True, False

        row2 = layout.row()
        prop2 = row2.operator(
            OBJECT_OT_AutoAlignOperator.bl_idname, text='Keep Position & Bake')
        prop2.bake, prop2.keep = True, True


classes = (
    OBJECT_OT_AutoAlignOperator,
    VIEW3D_PT_AutoAlignUi,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)


if __name__ == '__main__':
    register()
