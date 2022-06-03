import bpy
import random
import math
import json
from datetime import datetime 
import sys
import os
import importlib

dir = os.path.dirname(bpy.data.filepath)
dir = os.path.join(dir, '../../common/scripts/')

if not dir in sys.path:
    sys.path.append(dir)

import argsParser
import blenderConfig
import helper
importlib.reload(argsParser)
importlib.reload(blenderConfig)
importlib.reload(helper)

print(f'\n\n****** START ******\n')
blenderConfig.enableGPUs("CUDA")

parser = argsParser.ArgumentParserForBlender()

parser.add_argument('-minD', '--minimumDistance', type=float, default=-1, help='Minimum distance from the camera to the stand with digital instruments (meters)')
parser.add_argument('-maxD', '--maximumDistance', type=float, default=-1, help='Maximum distance from the camera to the stand with digital instruments (meters)')

args = parser.parse_args()

minD = args.minimumDistance
maxD = args.maximumDistance

def validateArgs():
    global minD
    if minD or minD == 0:
        if minD <= 0:
            minD = 3

    global maxD
    if maxD or maxD == 0:
        if maxD < 0:
            maxD = 8

    if minD > maxD:
        tempD = minD
        minD = maxD
        maxD = tempD 

validateArgs()

class RenderSettings:
    def __init__(self, scene, rootDir, datetimeFormat, imgsSubPaths: dict, masksSubPaths: dict):
        self.renderDir = helper.getRenderDir(rootDir, datetimeFormat)
        
        if not os.path.exists(self.renderDir):
            os.makedirs(self.renderDir)

        self.scene = scene
        self.annotationPath = os.path.join(self.renderDir, 'annotations.txt')
        self.imagePath = os.path.join(self.renderDir, 'images')
        self.imgsSubPaths = imgsSubPaths
        self.imgSubPath = 'img_'
        self.masksPath = os.path.join(self.renderDir, 'masks')
        self.masksSubPaths = masksSubPaths

        self.scene.render.filepath = f'{self.imagePath}/{self.imgSubPath}'

        print(f'RENDER OUTPUT: {self.scene.render.filepath}')

        for k, v in self.imgsSubPaths.items():
            self.scene.node_tree.nodes[k].base_path = self.imagePath
            self.scene.node_tree.nodes[k].file_slots[0].path = v

        for k, v in self.masksSubPaths.items():
            self.scene.node_tree.nodes[k].base_path = self.masksPath
            self.scene.node_tree.nodes[k].file_slots[0].path = v

class DigitalMeasurement:
    def __init__(self, obj, name: str, minValue: float, maxValue: float, valueLength: int, num: int, digitObjs: list):
        self.obj = obj
        self.name = name
        self.minValue = minValue
        self.maxValue = maxValue
        self.value = self.minValue
        self.num = num
        self.valueLength = valueLength
        self.valueStr = ''
        self.obj.data.body = self.valueStr
        self.digitObjs = digitObjs

    def valueToStr(self):
        valueStr = str(self.value)
        
        if self.num <= 0:
            valueStr = valueStr[0:len(valueStr)-2]
        
        print(f'valueStr: {valueStr}')

        while len(valueStr) < self.valueLength:
            valueStr = f' {valueStr}'


        return valueStr

    def updateValue(self):
        self.value = random.uniform(self.minValue, self.maxValue)
        print(f'value: {self.value}')
        self.value = round(self.value, self.num)
        self.valueStr = self.valueToStr()
        self.obj.data.body = self.valueStr

        self.writeDigitsBbox()

    def writeDigitsBbox(self):
        data = {}
        for k, v in self.digitObjs.items():
            cls = self.valueStr[-k]

            if cls == ' ':
                continue
            
            if cls == '.':
                cls = '10'

            data[f'bbox_0{k}'] = helper.bboxToDict(helper.camera_view_bounds_2d(C.scene, C.scene.camera, v), cls=cls)
        return data

class DigitalManometerStand:
    def __init__(self, obj, rs: RenderSettings, dms: list, xyz: list, minXYZ: list, maxXYZ: list, 
                 minAngleXYZ: list, maxAngleXYZ: list, minD: float, maxD: float):
        self.obj = obj
        self.rs = rs
        self.dms = dms
        self.xyz = xyz
        self.minXYZ = minXYZ
        self.maxXYZ = maxXYZ
        self.minAngleXYZ = minAngleXYZ
        self.maxAngleXYZ = maxAngleXYZ
        self.minD = minD
        self.maxD = maxD
        self.coefX = abs((self.maxXYZ[0] - self.minXYZ[0]) / (self.maxXYZ[1] - self.xyz[1]))
        self.coefZ = abs((self.maxXYZ[2] - self.minXYZ[2]) / (self.maxXYZ[1] - self.xyz[1]))

    def getRandLocation(self):
        y = random.uniform(self.minD, self.maxD)

        rangeX = abs((y-self.minD)*self.coefX)
        rangeZ = abs((y-self.minD)*self.coefZ)

        x = random.uniform(-rangeX, rangeX)
        z = random.uniform(-rangeZ, rangeZ)

        print(f'Stand Location: {x}, {y}, {z}')
        self.obj.location = [x, y, z]
        return [x, y, z]

    def getRandRotation(self):
        x = self.minAngleXYZ[0]+random.uniform(-self.maxAngleXYZ[0], self.maxAngleXYZ[0])
        y = self.minAngleXYZ[1]+random.uniform(-self.maxAngleXYZ[1], self.maxAngleXYZ[1])
        z = self.minAngleXYZ[2]+random.uniform(-self.maxAngleXYZ[2], self.maxAngleXYZ[2])

        print(f'Stand Rotation: {x}, {y}, {z}')

        x = math.radians(x)
        y = math.radians(y)
        z = math.radians(z)

        self.obj.rotation_euler = [x, y, z]
        return [x, y, z]

    def writeInfo(self, frame):
        data = {
            'frame': f'{frame}',
            'imagePath': self.rs.imagePath,
            'maskPath': self.rs.masksPath,
        }
        
        for k, v in self.rs.imgsSubPaths.items():
            data[v[:-1]] = f'{helper.getImgName(v, frame, ".png")}'

        for k, v in self.rs.masksSubPaths.items():
            data[v[:-1]] = f'{helper.getImgName(v, frame, ".png")}'

        for i in range(len(self.dms)):
            data[f'dm0{i+1}'] = self.dms[i].getAnnotation()

        if frame == 1:
            jsonToStr = f'[\n{json.dumps(data, indent = 4)}]'
            annotationsFile = open(self.rs.annotationPath, 'w+')
            annotationsFile.write(jsonToStr)
            annotationsFile.close()
        else:
            jsonToStr = f',\n{json.dumps(data, indent = 4)}]'
            annotationsFile = open(self.rs.annotationPath, 'r+')
            line = annotationsFile.read()
            annotationsFile.seek(annotationsFile.tell()-1)
            annotationsFile.write(jsonToStr)
            annotationsFile.close()

class DigitalManometer:
    def __init__(self, obj, displayObjs, measurements: list):
        self.obj = obj
        self.displayObjs = displayObjs
        self.measurements = measurements
        self.currentValues = {}

    def updateValues(self):
        for m in self.measurements:
            m.updateValue()

    def updateBbox(self, scene, camera):
        self.bbox = helper.camera_view_bounds_2d(scene, camera, self.obj)

    def updateDisplayBbox(self, scene, camera):
        self.displayBboxes = []
        for i in range(len(self.displayObjs)):
            self.displayBboxes.append(helper.camera_view_bounds_2d(scene, camera, self.displayObjs[i]))

    def getAnnotation(self):
        data = {
            'bbox': helper.bboxToDict(self.bbox),
            # 'displayBbox': helper.bboxToDict(self.displayBbox),
        }

        for i in range(len(self.displayBboxes)):
            data[f'displayBbox{i}'] = helper.bboxToDict(self.displayBboxes[i])

        for m in self.measurements:
            data[m.name] = m.value
            data[f'{m.name}Bboxes'] = m.writeDigitsBbox()


        return data

datetimeFormat = "%Y%m%d_%H%M%S"

C = bpy.context

bpy.app.handlers.frame_change_pre.clear()
C.scene.frame_set(0)

rootDir =  os.path.join(os.getcwd(), 'renders/')

rs = RenderSettings(C.scene, rootDir, datetimeFormat,
                        imgsSubPaths={}, 
                        masksSubPaths={'ManometerMaskOutput': 'mask01_', 'DisplayMaskOutput': 'mask02_'})

value = DigitalMeasurement(C.scene.objects[f'Numbers'], 'numbers', 0, 9999, valueLength=4, num=0,
                            digitObjs={1: C.scene.objects[f'Number4'], 2: C.scene.objects[f'Number3'], 3: C.scene.objects[f'Number2'], 4: C.scene.objects[f'Number1']})

dm = DigitalManometer(C.scene.objects[f'DigitalManometer'], [C.scene.objects[f'Display']], [value])

standObj = C.scene.objects[f'Stand']
stand = DigitalManometerStand(standObj, rs, [dm], 
                              xyz=[0, 3, -0.75], minXYZ=[0, 0, 0], maxXYZ=[1.8, 8.0, 2.25], 
                              minAngleXYZ=[0, 0, 0], maxAngleXYZ=[15, 15, 15], 
                              minD=minD, maxD=maxD)

envTextureMapping = C.scene.world.node_tree.nodes["Env Mapping"]
envTextureRotation = envTextureMapping.inputs['Rotation']

num = 1

D = bpy.data

global framesG
framesG = -1

def frame_change_handler(scene):
    print(f'Minimum Distance: {minD} meters')
    print(f'Maximum Distance: {maxD} meters')

    global framesG

    currentFrame = scene.frame_current
    if framesG == currentFrame or currentFrame == 0:
        return

    framesG = currentFrame
    currentFrame = scene.frame_current

    helper.setEnvRotation(envTextureRotation, 2)

    stand.getRandLocation()
    stand.getRandRotation()
    
    for i in range(len(stand.dms)):
        stand.dms[i].updateBbox(C.scene, C.scene.camera)
        stand.dms[i].updateDisplayBbox(C.scene, C.scene.camera)
        stand.dms[i].updateValues()

    stand.writeInfo(framesG)
    print(f'\n\n')


bpy.app.handlers.frame_change_pre.append(frame_change_handler)
bpy.context.scene.frame_set(0)

print(f'\n****** END ******')