from datetime import datetime 
import os
import math
import random
import bpy


def getRenderDir(rootDir: str, datetimeFormat: str):
    datetimeStr = datetime.now().strftime(datetimeFormat)
    renderDir = os.path.join(rootDir, f'render_{datetimeStr}')
    return renderDir


def getRandomRotation(minXYZ=[0, 0, 0], maxXYZ=[360, 360, 360]):
    x = math.radians(random.uniform(minXYZ[0], maxXYZ[0]))
    y = math.radians(random.uniform(minXYZ[1], maxXYZ[1]))
    z = math.radians(random.uniform(minXYZ[2], maxXYZ[2]))

    return (x, y, z)


def getImgName(name: str, frame: int, format: str):
    if frame < 10:
        return f'{name}000{frame}{format}'
    elif frame < 100:
        return f'{name}00{frame}{format}'
    elif frame < 1000:
        return f'{name}0{frame}{format}'
    else:
        return f'{name}{frame}{format}'


def bboxToDict(bbox, x=0, y=0, width=0, height=0, cls=None):
    data = {}
    
    if bbox:
        centerX = round(bbox[0] + bbox[2] / 2)
        centerY = round(bbox[1] + bbox[3] / 2)
        data = {'x': bbox[0], 'y': bbox[1], 'width': bbox[2], 'height': bbox[3], 'centerX': centerX, 'centerY': centerY}
    else:
        centerX = round(x + width / 2)
        centerY = round(y + height / 2)
        data = {'x': x, 'y': y, 'width': width, 'height': height, 'centerX': centerX, 'centerY': centerY}

    if cls != None:
        data[f'cls'] = cls

    return data


def clamp(x, minimum, maximum):
    return max(minimum, min(x, maximum))


def camera_view_bounds_2d(scene, cam_ob, me_ob):
    """
    Returns camera space bounding box of mesh object.
    
    Negative 'z' value means the point is behind the camera.

    Takes shift-x/y, lens angle and sensor size into account
    as well as perspective/ortho projections.

    :arg scene: Scene to use for frame size.
    :type scene: :class:`bpy.types.Scene`
    :arg obj: Camera object.
    :type obj: :class:`bpy.types.Object`
    :arg me: Untransformed Mesh.
    :type me: :class:`bpy.types.Mesh´
    :return: a Box object (call its to_tuple() method to get x, y, width and height)
    :rtype: :class:`Box`
    """

    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)

    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'

    lx = []
    ly = []

    for v in me.vertices:
        co_local = v.co
        z = -co_local.z

        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            # Does it make any sense to drop these?
            # if z <= 0.0:
            #    continue
            else:
                frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    mesh_eval.to_mesh_clear()

    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac

    # Sanity check
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0, 0, 0, 0)

    return (
        round(min_x * dim_x),            # X
        round(dim_y - max_y * dim_y),    # Y
        round((max_x - min_x) * dim_x),  # Width
        round((max_y - min_y) * dim_y)   # Height
    )


def setEnvRotation(envRotation, axis: int = -1):
    if axis == -1:
        return
    else:
        xyz = [0, 0, 0]
        xyz[axis] = 360
        envRotation.default_value = getRandomRotation(minXYZ=[0, 0, 0], maxXYZ=xyz)