import numpy as np
import bpy
from bpy.props import BoolProperty, PointerProperty
from bpy.types import Panel, Operator, PropertyGroup

bl_info = {
    'name': 'Auto Align',
    'author': 'cubec',
    'blender': (4, 0, 0),
    'version': (0, 6, 1),
    'category': 'Object',
    'description': 'Automatically re-aligns wrong axis objects',
    'doc_url': 'https://github.com/AlexsandroAF/Auto-Align-Blender-v4/blob/main/README.md'
}

# Hyperparameters
ITERATION_RANSAC = 200
ITERATION_MEDIAN = 10
THRESHOLD = 5 * (np.pi / 180)
MAX_POLYS = 10000
MAX_POLYS_SUBSET = 100
SYMMETRY_PAIR_DIST = 0.03
SYMMETRY_BUCKET_SIZE = 0.1


class AutoAlignProperties(PropertyGroup):
    symmetry: BoolProperty(default=False, name='Symmetry')


class OBJECT_OT_AutoAlignBaseOperator(Operator):
    bl_idname = 'object.auto_align_base'
    bl_label = 'Auto Align'
    bl_description = 'Automatically re-aligns wrong axis objects'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        auto_align = context.scene.auto_align
        align(context, symmetry=auto_align.symmetry)
        return {'FINISHED'}


class OBJECT_OT_AutoAlignBakeOperator(Operator):
    bl_idname = 'object.auto_align_bake'
    bl_label = 'Auto Align'
    bl_description = 'Automatically re-aligns wrong axis objects'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        auto_align = context.scene.auto_align
        align(context, bake=True, symmetry=auto_align.symmetry)
        return {'FINISHED'}


class OBJECT_OT_AutoAlignKeepOperator(Operator):
    bl_idname = 'object.auto_align_keep'
    bl_label = 'Auto Align'
    bl_description = 'Automatically re-aligns wrong axis objects'
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        auto_align = context.scene.auto_align
        align(context, keep=True, bake=True, symmetry=auto_align.symmetry)
        return {'FINISHED'}


def align(context, bake=False, keep=False, symmetry=False):
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
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

        if symmetry:
            verts = m.data.vertices
            normals_vert = np.array([list(v.normal)
                                     for v in verts]) @ global_matrix[:3, :3].T
            normals_vert = normals_vert / \
                np.linalg.norm(normals_vert, axis=1, keepdims=True)
            positions_vert = np.array([list(v.co)
                                       for v in verts]) @ global_matrix[:3, :3].T
            plane = get_symmetry_plane(normals_vert, positions_vert)
            model = get_matrix(areas, normals, fixed_axis=plane[:3])
        else:
            model = get_matrix(areas, normals)

        global_matrix[:3, :3] = model @ global_matrix[:3, :3]
        m.matrix_basis = global_matrix.T

        if keep:
            keep_bucket.append((m, model))

    if bake:
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=False)

    if keep:
        for m, model in keep_bucket:
            global_matrix = np.array(m.matrix_basis)
            global_matrix[:3, :3] = model.T @ global_matrix[:3, :3]
            m.matrix_basis = global_matrix.T


def get_symmetry_plane(normals, positions):
    if normals.shape[0] > MAX_POLYS:
        indices = np.random.choice(normals.shape[0], MAX_POLYS, replace=False)
        normals = normals[indices]
        positions = positions[indices]

    if normals.shape[0] > MAX_POLYS_SUBSET:
        indices = np.random.choice(normals.shape[0], MAX_POLYS_SUBSET, replace=False)
        normals_subset = normals[indices]
        positions_subset = positions[indices]
    else:
        normals_subset = normals
        positions_subset = positions

    positions_1 = np.tile(positions, (normals_subset.shape[0], 1))
    positions_2 = np.repeat(positions_subset, normals.shape[0], axis=0)
    normals_1 = np.tile(normals, (normals_subset.shape[0], 1))
    normals_2 = np.repeat(normals_subset, normals.shape[0], axis=0)
    plane_normals = positions_1 - positions_2
    plane_normals_scale = np.linalg.norm(plane_normals, axis=1)
    plane_normals /= (plane_normals_scale + 1e-6).reshape(-1, 1)
    normals_3 = normals_1 - 2 * plane_normals * \
        np.sum(plane_normals * normals_1, axis=1).reshape(-1, 1)

    indices = np.nonzero((np.linalg.norm(normals_2 - normals_3, axis=1)
                          < SYMMETRY_PAIR_DIST) & (plane_normals_scale > 1e-6))[0]
    plane_normals = plane_normals[indices]
    plane_centers = np.sum((positions_1 + positions_2)[indices] / 2 * plane_normals, axis=1)

    plane = np.concatenate(
        (plane_normals, plane_centers.reshape(-1, 1)), axis=1)
    plane_centers_std = np.std(plane[:, 3])
    plane[:, 3] /= (plane_centers_std + 1e-6)

    plane_int = np.rint(plane / SYMMETRY_BUCKET_SIZE).astype(int)
    plane_range = np.ptp(plane_int, axis=0) + 1
    plane_int_hash = plane_int[:, 0] + plane_int[:, 1] * plane_range[0] \
        + plane_int[:, 2] * plane_range[0] * plane_range[1] \
        + plane_int[:, 3] * plane_range[0] * plane_range[1] * plane_range[2]
    value, count = np.unique(plane_int_hash, return_counts=True)
    origin = plane_int[(plane_int_hash == value[np.argmax(count)]).nonzero()[0][0]] * SYMMETRY_BUCKET_SIZE
    dist = np.linalg.norm(plane - origin.reshape(1, -1), axis=1)
    plane_res = np.median(plane[(dist < SYMMETRY_BUCKET_SIZE).nonzero()[0]], axis=0)
    plane_res[3] *= (plane_centers_std + 1e-6)
    plane_res[:3] /= np.linalg.norm(plane_res[:3])

    return plane_res


def get_matrix(areas, normals, fixed_axis=None):
    if areas.size > MAX_POLYS:
        indices = np.random.choice(areas.size, MAX_POLYS, p=areas / sum(areas), replace=False)
        areas = areas[indices]
        normals = normals[indices]

    first_indices = np.random.choice(areas.size, ITERATION_RANSAC, p=areas / sum(areas))
    best_model = np.identity(3)
    best_value = -1.0

    for index in first_indices:
        model = np.zeros((3, 3))
        if fixed_axis is None:
            model[0] = normals[index]
        else:
            model[0] = fixed_axis

        next_indices = np.nonzero(np.abs(normals @ model[0]) < np.sin(THRESHOLD))[0]
        if next_indices.size > 0:
            next_areas = areas[next_indices]
            model[1] = normals[np.random.choice(next_indices, p=next_areas / sum(next_areas))]
        else:
            model[1] = np.zeros(3)
            model[1][(np.argmax(np.abs(model[0])) + 1) % 3] = 1

        model[1] = np.cross(model[0], model[1])
        model[1] /= np.linalg.norm(model[1])
        model[2] = np.cross(model[0], model[1])

        indices = np.max(np.abs(normals @ model.T), axis=1) > np.cos(THRESHOLD)
        value = np.sum(areas[indices])
        if best_value < value:
            best_value, best_model, best_indices = value, model, indices

    return best_model


class VIEW3D_PT_AutoAlignUi(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_label = 'Auto Align'
    bl_context = 'objectmode'
    bl_category = 'Item'
    bl_options = {'DEFAULT_CLOSED'}

    def draw(self, context):
        layout = self.layout
        auto_align = context.scene.auto_align

        layout.row().prop(auto_align, 'symmetry', text='Symmetry')
        row = layout.column_flow(columns=1, align=True)
        row.operator(OBJECT_OT_AutoAlignBaseOperator.bl_idname, text='Rotate')
        row.operator(OBJECT_OT_AutoAlignBakeOperator.bl_idname, text='Rotate & Bake')
        row.operator(OBJECT_OT_AutoAlignKeepOperator.bl_idname, text='Keep Position & Bake')


classes = (
    AutoAlignProperties,
    OBJECT_OT_AutoAlignBaseOperator,
    OBJECT_OT_AutoAlignBakeOperator,
    OBJECT_OT_AutoAlignKeepOperator,
    VIEW3D_PT_AutoAlignUi,
)


def register():
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.auto_align = PointerProperty(type=AutoAlignProperties)


def unregister():
    for cls in classes:
        bpy.utils.unregister_class(cls)

    del bpy.types.Scene.auto_align


if __name__ == '__main__':
    register()
