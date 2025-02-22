#!/usr/bin/env python
# encoding: utf-8
import os
import warnings
import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from skimage.measure import label, regionprops
from tqdm import tqdm
from lxml import etree
from io import *



class Vesicle:

    """
    parameters(radius, distance, position etc.) stores in pixel. If you want to use nm, please multiple self.getPixelSize
    TODO:elliptical vesicle
    """
    def __init__(self, recFile = None):
        self._vesicleId = 0


    def fromXML(self, xmlObj, pixelSize, isPytomFormat = False):
        if isPytomFormat:
            self.fromXMLPytom(xmlObj, pixelSize)
        else:
            self.fromXMLSynTomo(xmlObj, pixelSize)


    def fromXMLPytom(self, xmlObj, pixelSize):
        """from XML in pytom format
        class name in xml is diameter of vesicle in nm.
        self._radius is radius in pixel
        @param xmlObj:A xml object
        """
        vesicleElement = xmlObj
        self._vesicleId = int(vesicleElement.get('Filename'))

        positionElement = vesicleElement.xpath('PickPosition')[0]

        self._center = [float(positionElement.get('X')),
                        float(positionElement.get('Y')),
                        float(positionElement.get('Z'))]


    def fromXMLSynTomo(self, xmlObj, pixelSize):
        """from XMl in synTomo format
        """
        vesicleElement = xmlObj
        self._vesicleId = int(vesicleElement.get("vesicleId"))
        argList = ["Radius2D", 
                   "Radius3D",
                   "Radius", 
                   "Rotation2D", 
                   "Center", 
                   "Center2D", 
                   "Center3D", 
                   "Distance", 
                   "ProjectionPoint", 
                   "Type",
                   "PitPoint"]  # ls: 添加 PitPoint 到参数列表
        argStrList = ["Type"]
        
        for item in argList:
            itemPath = vesicleElement.xpath(item)

            if len(itemPath) > 0:
                its = itemPath[0].items()

                if item in argStrList:
                    setattr(self, f"_{item[0].lower()}{item[1:]}", its[0][1])
                elif item == "PitPoint":  # ls: 处理 PitPoint
                    setattr(self, f"_{item[0].lower()}{item[1:]}", [
                        float(its[0][1]),
                        float(its[1][1]),
                        float(its[2][1])
                    ])
                else:
                    array = np.array([float(val[1]) for val in its], dtype=float)
                    if len(array) == 1:
                        array = array[0]
                    setattr(self, f"_{item[0].lower()}{item[1:]}", array)
        
        if len(vesicleElement.xpath("Evecs")) > 0:
            evecs = np.zeros((3, 3), dtype=float)
            for evec in vesicleElement.xpath("Evecs"):
                idx = int(evec.get("idx"))
                evec_values = [float(evec.get(coord)) for coord in ["X", "Y", "Z"]]
                evecs[idx, :] = evec_values
            self._evecs = evecs

        # ls: 解析 PitPoint 在 SynTomo 格式中
        pit_point_elements = vesicleElement.xpath('PitPoint')
        if pit_point_elements:
            pit_point = pit_point_elements[0]
            self._pitPoint = [
                float(pit_point.get('X')),
                float(pit_point.get('Y')),
                float(pit_point.get('Z'))
            ]


    def toXML(self, pixelSize):
        """
        @param pixelSize: the pixel size in tomogram in nm/pixel
        """
        vesicleElement = etree.Element("Vesicle",\
                                       vesicleId = str(self._vesicleId))
        if hasattr(self,"_type"):
            vesicleElement.append(etree.Element("Type",\
                                                t=str(self._type)))

        if hasattr(self, "_center"):
            vesicleElement.append(etree.Element("Center",\
                                                X = str(self._center[0]),\
                                                Y = str(self._center[1]),\
                                                Z = str(self._center[2] )))
        if hasattr(self,"_radius"):
            vesicleElement.append(etree.Element("Radius",\
                                                r = str(self._radius)))
        if hasattr(self,"_center2D"):
            vesicleElement.append(etree.Element("Center2D",\
                                                X = str(self._center2D[0]),\
                                                Y = str(self._center2D[1]),\
                                                Z = str(self._center2D[2])))
        if hasattr(self, "_center3D"):
            vesicleElement.append(etree.Element("Center3D",\
                                                X = str(self._center3D[0]),\
                                                Y = str(self._center3D[1]),\
                                                Z = str(self._center3D[2])))
        if hasattr(self,"_radius2D"):
            vesicleElement.append(etree.Element("Radius2D",\
                                                r1 = str(self._radius2D[0]),\
                                                r2 = str(self._radius2D[1])))
        if hasattr(self,"_radius3D"):
            vesicleElement.append(etree.Element("Radius3D",\
                                                r1 = str(self._radius3D[0]),\
                                                r2 = str(self._radius3D[1]),\
                                                r3 = str(self._radius3D[2])))
        if hasattr(self,"_rotation2D"):
            vesicleElement.append(etree.Element("Rotation2D",\
                                                phi = str(self._rotation2D)))
        
        if hasattr(self,"_evecs"):
            for i, evec in enumerate(self._evecs.T):
                vesicleElement.append(etree.Element("Evecs",\
                                                    X = str(evec[2]),\
                                                    Y = str(evec[1]),\
                                                    Z = str(evec[0]),\
                                                    idx = str(i)))

        if hasattr(self,"_distance"):
            vesicleElement.append(etree.Element("Distance",\
                                                d = str(self._distance)))

        if hasattr(self,"_projectionPoint"):
            vesicleElement.append(etree.Element("ProjectionPoint",\
                                                X = str(self._projectionPoint[0]),\
                                                Y = str(self._projectionPoint[1]),\
                                                Z = str(self._projectionPoint[2])))
        # ls: 添加 PitPoint 元素
        if hasattr(self, "_pitPoint"):
            vesicleElement.append(etree.Element("PitPoint",\
                                                X = str(self._pitPoint[0]),\
                                                Y = str(self._pitPoint[1]),\
                                                Z = str(self._pitPoint[2])))

        return vesicleElement


    def ellipsoid_equation(self):
        '''
        to get the equation of a 3D ellipsoid from radii and directions
        '''
        D_inv2 = np.diag(1 / np.array(self._radius3D)**2)
        U = np.column_stack(self._evecs)
        A = U @ D_inv2 @ U.T
        
        return A


    def ellipse_in_plane(self):
        '''
        get the parameters of the 2D ellipse parallel to the xy-plane and the center of the 3D ellipsoid
        '''
        A = self.ellipsoid_equation()
        A_2d = A[:2, :2]
        eigvals, eigvecs = np.linalg.eigh(A_2d)
        axes_lengths = np.sqrt(1 / eigvals)

        return self._center3D, axes_lengths, eigvecs


    def largest_cross_section(self):
        '''
        get the parameters of the max 2D ellipse of the 3D ellipsoid (which is exactly the ellipse composed of the major and the second major axes)
        NOT recommended. Not fit with other 2D data.
        '''
        A = self.ellipsoid_equation()
        eigvals, eigvecs = np.linalg.eigh(A)
        max_eigval_index = np.argmax(eigvals)
        axes_lengths = np.sqrt(1 / eigvals[np.delete(np.arange(3), max_eigval_index)])
        rotation_matrix = eigvecs[:, np.delete(np.arange(3), max_eigval_index)]

        return self._center3D, axes_lengths, rotation_matrix


    def distance_to_surface(self, surface, precision, tree, membrane_points):
        """
        @param surface: membrane.surface instance
        @author: Lu Zhenhang
        @date: 2024-10-14
        """
        
        if hasattr(self, "_center3D"):
            
            points = self.sample_on_vesicle_3d_fibonacci(precision)
            dist, idx = tree.query(points, k=1)
            dis = np.min(dist)
            fit_PP0_idx = idx[np.argmin(dist)]
            nearest_point = points[np.argmin(dist)]
            PP0 = membrane_points[fit_PP0_idx[0]]
            
        elif hasattr(self, "_center2D"):
            
            points = self.sample_on_vesicle(precision)
            dist, idx = tree.query(points, k=1)
            dis = np.min(dist)
            fit_PP0_idx = idx[np.argmin(dist)]
            nearest_point = points[np.argmin(dist)]
            PP0 = membrane_points[fit_PP0_idx[0]]

        self._distance = dis
        self._projectionPoint = PP0

        return dis, PP0, nearest_point


    def sample_on_vesicle(self, precision : int) -> np.ndarray:
        """
        @param precision: number of points sampled on vesicle (2d max section)
        @return: points list sampled on a vesicle
        @author: Lu Zhenhang
        @date: 2023-04-18
        """
        """
        -> fix bugs on 230604 by Lu Zhenhang, the self._rotation2D starts from -pi/2, and it belongs to the first axis, not the major axis"""
        
        phi = np.float64(self._rotation2D) + np.pi/2
        a = np.float64(self._radius2D[0])
        b = np.float64(self._radius2D[1])
        
        # use the parametric equation of an ellipse
        x = a * np.cos(np.linspace(0, 2 * np.pi, precision)) * np.cos(phi) - b * np.sin(np.linspace(0, 2 * np.pi, precision)) * np.sin(phi)
        y = a * np.cos(np.linspace(0, 2 * np.pi, precision)) * np.sin(phi) + b * np.sin(np.linspace(0, 2 * np.pi, precision)) * np.cos(phi)
        z = np.ones(precision)
        points = np.vstack((x, y, z)).T + self._center2D
        
        # assert points.shape == (precision, 3), f"Unexpected shape: {points.shape}"
        return points
    
    
    def sample_on_vesicle_3d_fibonacci(self, precision : int) -> np.ndarray:
        '''
        to get uniform points on the surface of a 3D ellipsoid
        
        C: center of ellipsoid: (3, )
        R: radii of ellipsoid: (3, )
        U: directions of ellipsoid: (3, 3)
        
        @author: Lu Zhenhang
        @date: 2024-10-14
        '''
        
        indices = np.arange(0, precision, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/precision)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        points = np.stack([x, y, z], axis=-1)  # (precision, 3), points is an isotropic sphere
        
        points *= self._radius3D
        points = points @ self._evecs + self._center3D

        # points = points[:, [2, 1, 0]]  # ls: dont need to change zyx to xyz 
        # assert points.shape == (precision, 3), f"Unexpected shape: {points.shape}"        
        return points
    
    
    def sample_on_vesicle_3d(self, precision : int) -> np.ndarray:
        '''
        to get random points on the surface of a 3D ellipsoid. For distance calculating, waste of points num.
        
        C: center of ellipsoid: (3, )
        R: radii of ellipsoid: (3, )
        U: directions of ellipsoid: (3, 3)
        
        @author: Lu Zhenhang
        @date: 2024-10-14
        '''
        
        random_points = np.random.normal(size=(precision, 3))
        random_points /= np.linalg.norm(random_points, axis=-1, keepdims=True)
        
        points = random_points * self._radius3D
        points = points @ self._evecs + self._center3D
        points = points[:, [2, 1, 0]]  # zyx to xyz

        # assert points.shape == (precision, 3), f"Unexpected shape: {points.shape}"
        return points
    
    
    def getCenter(self):
        if hasattr(self,'_center3D'):
            return self._center3D
        elif hasattr(self,'_center2D'):
            return self._center2D
        else:
            return self._center

    def setCenter(self, center):
        self._center = center
        self._center2D = center
        self._center3D = center

    def setId(self, Id):
        self._vesicleId = Id

    def getRadius(self):
        if hasattr(self, '_radius3D'):
            return self._radius3D
        elif hasattr(self, '_radius2D'):
            return self._radius2D
        return self._radius

    def getRadius2D(self):
        return self._radius2D
    
    def getRadius3D(self):
        warnings.warn("Vesicle().getRadius3D() is deprecated and will be removed soon, please use getRadius() instead", UserWarning)
        return self._radius3D

    def setRadius(self, radius):
        self._radius = radius
    
    def setRadius2D(self, radius2D):
        self._radius2D = radius2D
    
    def setRadius3D(self, radius3D):
        self._radius3D = radius3D
    
    def setEvecs(self, evecs):
        self._evecs = evecs
    
    def getEvecs(self):
        return self._evecs

    def getId(self):
        return self._vesicleId

    def getProjectionPoint(self):
        return self._projectionPoint

    def setProjectionPoint(self, projectionPoint):
        self._projectionPoint = projectionPoint

    def setRotation2D(self, Rotation2D):
        self._rotation2D = Rotation2D

    def getRotation2D(self):
        return self._rotation2D

    def setPitPoint(self, pitPoint):
        self._pitPoint = pitPoint

    def getPitPoint(self):
        return self._pitPoint
    
    def getDistance(self):
        return self._distance

    def setDistance(self,d):
        self._distance = d

    def getType(self):
        return self._type
    
    def setType(self,t):
        self._type = t


from typing import List, Iterator
class VesicleList:
    """parameters(radius, distance, position etc.) stores in pixel. If you want to use nm, please multiply self.getPixelSize
    """
    def __init__(self, pixelSize = 1.0):
        self._vesicleList: List[Vesicle] = []
        self._pixelSize = pixelSize

    def __len__(self):
        return len(self._vesicleList)

    def __setitem__(self, key, value):
        self._vesicleList[key]=value

    def __getitem__(self, key) -> Vesicle:
        """retrive vesicle at position defined by key
        """

        if isinstance(key, int):
            if key < len(self):
                return self._vesicleList[key]
            else:
                raise IndexError('Index out of range')
    
    def __iter__(self) -> Iterator[Vesicle]:
        return iter(self._vesicleList)

    def __add__(self, vesicleList):
        """
        Concatenates two ParticleLists
        @param vesicleList
        """
        if not vesicleList.__class__ == VesicleList:
            raise TypeError("Can not concatenate this vesicleList to a non VesicleList object")

        beginId = self._vesicleList[-1].getId() + 1
        for i,vesicle in enumerate(vesicleList):
            self._vesicleList.append(vesicle)
            self._vesicleList[-1].setId(beginId + i)

        return self

    def append(self, vesicle):
        """
        append vesicle to self._vesicleList
        """
        assert vesicle.__class__ == Vesicle
        self._vesicleList.append(vesicle)


    def get_vesicle_by_center(self, position, distanceRange = 3.0):
        """Give a position, find corresponding vesicle whose center is in a distance range
        @param position: position near center
        @type position: instance of Point or a list with number 3 elements
        @param distanceRange: if the distance from position to center of a vesicle return this vesicle
        @type distanceRange: int or float
        @return: The corresponding vesicle
        """
        for vesicle in self._vesicleList:
        #    print vesicle.getCenter() - position
            if np.linalg.norm(position - vesicle.getCenter()) < distanceRange:
                return vesicle

        return False


    def get_vesicle_by_Id(self, Id):
        for vesicle in self._vesicleList:
            if str(vesicle.getId()) == str(Id):
                return vesicle
        print("No vesicle of given Id")
        return False


    def fromXMLFile(self, xmlFile, isPytomFormat = False):

        with open(xmlFile, 'r') as f:
            string = f.read()
            f.close()
        root = etree.fromstring(string)
        self.fromXML(root, isPytomFormat)


    def fromXML(self, xmlObj, isPytomFormat = False):
        """get vesicles from vesicleList generated by pytom"""
        directoryElement = xmlObj

        if not isPytomFormat:
            vesicles = directoryElement.xpath('Vesicle')
            self._pixelSize = float(directoryElement.get('pixelSize'))
        else:
            vesicles = directoryElement.xpath('Particle')

        if not hasattr(self, '_pixelSize'):
            raise ValueError("pixel size should be set in advance")

        if len(vesicles) > 0:
            for p in vesicles:
                pp = Vesicle()
                pp.fromXML(p, self._pixelSize, isPytomFormat)
                self._vesicleList.append(pp)


    def toXML(self):
        rootTree = etree.Element("VesicleList", pixelSize = str(self._pixelSize))
        for vesicle in self._vesicleList:
            rootTree.append(vesicle.toXML(self._pixelSize))
        return rootTree


    def toXMLFile(self, outputXMLFile):
        # ls
        output_dir = os.path.dirname(outputXMLFile)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        xmlString = etree.tostring(self.toXML(), pretty_print = True).decode('utf-8')
        with open(outputXMLFile, 'w') as f:
            f.write(xmlString)
            f.close()


    def fromCenterList(self, centerList):
        """initialize the instance with center list (n*3),
        creat n vesicle instances in self._vesicleList
        set the center and id of vesicles
        @param centerList:"""
        if not isinstance(centerList, np.ndarray):
            centerList = np.array(centerList, dtype = float)
        self._centerList = centerList
        self._vesicleList = []
        for i,center in enumerate(centerList):
            vesicle = Vesicle()
            vesicle.setCenter(center)
            vesicle.setId(i+1)
            self._vesicleList.append(vesicle)
    
    
    def distance_to_surface(self, surface, precision, mode='dense'):
        """
        -> rewrite by Lu Zhenhang, 2023-04-18. Using kd-tree to speed up the calculation
        -> rewrite by Lu Zhenhang, 2023-06-04. Add projection_point to output file
        
        -> update by Lu Zhenhang, 2024-10-14. Add 3D ellipsoid support (data from DL-based segmentation)
        @new param: mode: 'sparse' or 'dense', representing manual segmentation or auto-segmentation for membrane
        """
        
        self._distance = []
        self._projectionPoint = []
        nearest_point_list = []
        modtxtFile = []

        # kd-tree construction
        if mode == 'dense':
            tree = KDTree(surface._densePoints, leaf_size=2)
            
        elif mode == 'sparse':
            sample_triangle_list = np.asarray([0,0,0])
            for i, t in enumerate(surface._triangleList):
                sample_triangle_list = np.vstack((sample_triangle_list, np.asarray(t.sampling_triangle()).astype(np.float64)))
            sample_triangle_arr = np.asarray(sample_triangle_list)[1:,:]
            surface._densePoints = sample_triangle_arr
            print('{} points are sampled on the premembrane surface'.format(sample_triangle_arr.shape[0]))
            tree = KDTree(sample_triangle_arr, leaf_size=2)

        # distance calculation
        for i,vesicle in tqdm(enumerate(self._vesicleList), dynamic_ncols=True, mininterval=0.5):
            # for pits defined by three points, set distance to 0 and projection point is the center
            if vesicle.getType() == 'pit':
                vesicle._distance = 0.
                vesicle._projectionPoint = vesicle.getCenter()
                self._distance.append(0)
                self._projectionPoint.append(vesicle.getCenter())
                nearest_point_list.append(vesicle.getCenter())
                modtxtFile.append(np.concatenate((np.array([1, i+1]), vesicle.getCenter())))
                modtxtFile.append(np.concatenate((np.array([1, i+1]), vesicle.getCenter())))

            # elif vesicle.getType() == 'vesicle':
            else:
                dis, PP0, nearest_point = vesicle.distance_to_surface(surface, precision, tree, surface._densePoints)
                self._distance.append(dis)
                self._projectionPoint.append(PP0)
                nearest_point_list.append(nearest_point)
                modtxtFile.append(np.concatenate((np.array([1, i+1]), nearest_point)))
                modtxtFile.append(np.concatenate((np.array([1, i+1]), PP0)))
        
        '''
        nearest_point.mod contains n contours, where n equals to the number of vesicles.
        Each contour contains 2 points on the vesicle and the membrane, respectively.
        Just to check the correctness of the distance calculation.
        '''
        # np.savetxt('nearest_point.txt', np.reshape(modtxtFile, (-1, 5)), fmt='%d')
        # cmd = 'point2model -sp 10 nearest_point.txt nearest_point.mod'
        # os.system(cmd)
        # return self._distance, self._projectionPoint


    def setPixelSize(self, pixelSize):
        self._pixelSize = pixelSize

    def getPixelSize(self):
        return self._pixelSize

    def getCenterList(self):
        """
        @return: a numpy array of all the center of vesicles in this vesicleList
        dtype = float means dtype = numpy.float64
        """
        self._centerList = []
        for vesicle in self._vesicleList:
            self._centerList.append(vesicle.getCenter())

        self._centerList = np.array(self._centerList, dtype = float)

        return self._centerList

    def getRadius(self):
        return self._radius

    def setRadius(self, radius):
        self._radius = radius

    def setRadius2D(self, radius2D):
        self._radius2D = radius2D
    
    def setRadius3D(self, radius3D):
        self._radius3D = radius3D

    def getRadius2D(self):
        return self._radius2D
    
    def getRadius3D(self):
        return self._radius3D


class Triangle:

    def __init__(self, p1, p2, p3):
        import numpy as np
        self.p1 = np.array(p1, dtype = np.float64)
        self.p2 = np.array(p2, dtype = np.float64)
        self.p3 = np.array(p3, dtype = np.float64)


    def center(self):
        return (self.p1+self.p2+self.p3)/3.0


    def point_triangle_distance(self, p):
        """calculate distance from a point to the triangle.
        @param p: a list of 3 floats
        @param standard: use the standard way to measure distance.
        if not, use sampling points in the triangle to estamate the distance
        if not, requires the triangle be sampled by self.sampling_triangle()
        @return distance, nearest point:
        """
        return self.point_triangle_distance_using_sample_points(p)


    def point_triangle_distance_using_sample_points(self, p):
        """calculate distance from a point to the triangle, triangle represented by a set of points filling triangle

        requires self.sampling_triangle
        """
        if hasattr(self, 'points') == False:
            self.sampling_triangle()
        import numpy as np
        distance = np.linalg.norm(p - self.points[0])
        PP0 = self.points[0]

        for pt in self.points[1:]:
            dis = np.linalg.norm(p - pt)
            if dis < distance:
                distance = dis
                PP0 = pt
        return distance, PP0


    def sampling_triangle(self, samplingStep = 1.0):
        """generate a set of points filling the triangle,
        assign values to self.points
        It is better self.p1 is the cross over point of  longest and middle edge of the triangle
        @param samplingStep: distance between points
        @return: a list of points
        @Auther: Liu Yuntao
        """
        import numpy as np
        a = self.p2 - self.p1
        b = self.p3 - self.p1 #need test
        samplingNum = np.array(list(map(np.linalg.norm,[a,b]))) / samplingStep
        aVector = a / samplingNum[0]
        bVector = b / samplingNum[1]

        # To make the other direction uniform
        dotab = np.dot(aVector, bVector)/ np.linalg.norm(aVector) / np.linalg.norm(bVector)
        theta = np.arccos(dotab)
        p = int(0.5 / np.tan(theta/2.0))

        # i/Num0 + j/Num1 < 1
        k = samplingNum[1]/samplingNum[0]
        points = []
        for i in np.arange(0, samplingNum[0]):
            for j in np.arange(0, samplingNum[1]-i*k):
                if p > 1:
                    if int(i-j)%p ==0:
                        pt = self.p1 + i*aVector + j*bVector
                        points.append(pt)
                else:
                        pt = self.p1 + i*aVector + j*bVector
                        points.append(pt)
        self.points = points
        return points

    
    def area(self):
        '''
        Comparing with Heron formula, cross product is more stable and efficient in 3d space.
        '''
        AB = self.p2 - self.p1
        AC = self.p3 - self.p1
        cross_product = np.cross(AB, AC)
        return 0.5 * np.linalg.norm(cross_product)
    
    
    def to_points(self) -> np.ndarray:
        '''
        output: vertices of the triangle, shape in (3, 3)
        '''
        return np.array([self.p1, self.p2, self.p3])


class Surface:
    '''
    Representation of a membrane. Methods of this class are used to **load, triangulate and calculate basic properties** of the membrane.
    Data could be loaded from 2 sources: manual segmentation or auto-segmentation.
    
    Generally, original points from manual segmentation are sparse, while those from auto-segmentation are dense.
    For manual segmentation, the original result is a sparse set of points, which should be dense-sampling after triangulating (this process will done automatically when the membrane loaded).
    For auto-segmentation result, this class could directly load the dense points.
    '''

    def __init__(self):
        self._vertices = []
        self._faces = []
        self._triangleList = []


    def _make_triangle_list(self):
        """generate triangleList from vertices and faces

        """
        for face in self._faces:
            self._triangleList.append(Triangle(self._vertices[face[0]],
                                               self._vertices[face[1]],
                                               self._vertices[face[2]]))


    def from_model_use_imod_mesh(self, model, objNum, outputVRML = "tmp.wrl"):
        """Use imodmesh to generate surface

        @param model: imod model file
        @param objNum: index of object, starts from 1
        @return: Nothing
        """
        #from synTomo.files.modelhandler import ImodModel
        #model = ImodModel(model)
        #model.toModelFile("tmp.mod", objNum, closedContour = False)
        from subprocess import call
        s = 'imodmesh  -sP {0} >> /dev/null; imod2vrml2 {0} {1} >> /dev/null'.format(model, outputVRML)
        call(s, shell = True)
        self.fromVrml2(outputVRML)
        return 0


    def fromVrml2(self, wrlFile):
        """initialize class from vrml file

        @return: Nothing
        """
        with open(wrlFile) as f:
            sVertices = False
            sFaces = False
            for line in f:
                if (sVertices == True) and (']' in line):
                    sVertices = False
                if sVertices:
                    self._vertices.append(list(map(float,line.split(",")[0].split())))
                if 'point [' in line:
                    sVertices = True

                if (sFaces == True) and (']' in line):
                    sFaces = False
                if sFaces:
                    self._faces.append(list(map(int,line.split(",")[0:3])))
                if 'coordIndex [' in line:
                    sFaces = True
        self._make_triangle_list()
        return 0
    
    
    def from_model_auto_segment(self, model, objNum, amp=1):
        '''
        load membrane from model file produced by auto-segmentation
        premembrane: object num: 2
        postmembrane: object num: 3
        '''
        
        def custom_round(x, base=0.5):
            '''
            set coordinate to the nearest 0.5
            '''
            return np.round(x / base) * base
        

        def avg_for_1d(idxs, length):
            '''
            average for 1d slice of x(or y)
            '''
            arr = np.zeros((length, )).astype(np.int16)
            idxs_from0 = (np.array(idxs) - min(idxs)).astype(np.int16)
            arr[idxs_from0] = 1
            lbl = label(arr)
            regions = regionprops(np.stack([lbl, lbl]))
            idxs_mean = []
            for r in regions:
                rx = r.centroid
                idxs_mean.append(custom_round(rx[1] + min(idxs)))
            
            return idxs_mean


        def max_filter(unfiltered):
            '''
            to fix conflicts that in the same contour, points with the same x(or y) have different y(or x) (here just do a adjusted NMS by mean y(or x))
            '''
            filtered = []
            contours = []
            obj, _, _, _, _ = unfiltered[0]
            for z in sorted(list(set(unfiltered[:, -1].tolist()))):
                contours.append(unfiltered[unfiltered[:, -1] == z])
            
            for i, contour in enumerate(contours):
                idx = i + 1
                point_x_set = sorted(list(set(contour[:, 2].tolist())))
                point_y_set = sorted(list(set(contour[:, 3].tolist())))
                x_diff = max(point_x_set) - min(point_x_set)
                y_diff = max(point_y_set) - min(point_y_set)
                if x_diff < y_diff:  # membrane is vertical, so average along the x axis
                    length = int(x_diff + 5)
                    for y in point_y_set:
                        point_equ_y = contour[contour[:, 3] == y]
                        obj, _, _, y, z = point_equ_y[0]
                        idxs = point_equ_y[:, 2]
                        idxs_mean = avg_for_1d(idxs, length)
                        for x in idxs_mean:
                            filtered.append([obj, idx, x, y, z])
                            
                else:  # membrane is horizontal, so average along the y axis
                    length = int(y_diff + 5)
                    for x in point_x_set:
                        point_equ_x = contour[contour[:, 2] == x]
                        obj, _, x, _, z = point_equ_x[0]
                        idxs = point_equ_x[:, 3]
                        idxs_mean = avg_for_1d(idxs, length)
                        for y in idxs_mean:
                            filtered.append([obj, idx, x, y, z])

            filtered = np.array(filtered)
            
            return filtered
        
        
        cmd = 'model2point -ob {} {} >> /dev/null'.format(model, model.replace('.mod', '.point'))
        os.system(cmd)
        untreated = np.loadtxt(model.replace('.mod', '.point'))
        untreated = untreated[untreated[:, 0] == objNum]
        
        membrane = max_filter(untreated)
        membrane = membrane * amp  # amplify the membrane to the original size
        np.savetxt(model.replace('.mod', '_filter.point'), membrane, fmt='%d %d %.2f %.2f %.2f')
        
        self._densePoints = membrane[:, 2:]
        self._make_triangle_list_denseInput()


    def sampling_triangles(self, samplingStep):
        """Get points in side all the triangles in the surface
        @param samplingStep: the interval of the adjecent sampling points, in pixels"""
        for t in self._triangleList:
            t.sampling_triangle(samplingStep)


    def surface_area(self):
        """Calculate area of the surface
        @return: surface area in pixel^2

        """
        area = 0
        for t in self._triangleList:
            area += t.area()
        return area


    def center(self, onSurf = False):
        """ if onSurf == True, the center is located on the surface
        """
        totalWeight = 0.0
        center = []
        for triangle in self._triangleList:
            if len(center) == 0:
                center = triangle.center()*triangle.area()
            else:
                center += triangle.center()*triangle.area()
            totalWeight += triangle.area()

        center = center / totalWeight

        if onSurf:
            dis,PP0 = self.point_distance(center)
            center = PP0

        return center
    
    
    def setPoints(self, points : np.ndarray):
        '''
        For auto-segmentation result, load dense points directly
        For load membrane from scratch, `from_model_auto_segment` method is recommended, which will load membrane from the original model file with no need for other process.
        '''
        self._densePoints = points
        self._make_triangle_list_denseInput()
    
    
    def getPoints(self) -> np.ndarray:
        return self._densePoints
    
    
    def _make_triangle_list_denseInput(self):
        '''
        For auto-segmentation result, generate triangle list from direct input dense points.
        Delaunay triangulation is used here. It works well when points are 2d, so projection is needed.
        
        For nearly all membrane of our synapse data, the points form a surface that is almost perpendicular to the xy-plane. 
        If projected directly onto the xy-plane, the points overlap significantly.
        
        Here we use PCA to get the normal vector of the surface, and project the points onto a plane perpendicular to the normal vector.
        '''
        
        pca = PCA(n_components=2, random_state=0)
        points_transformed = pca.fit_transform(self._densePoints)
        
        delaunay = Delaunay(points_transformed)
        
        for simplex in delaunay.simplices:
            # use original coordinates here
            p1 = self._densePoints[simplex[0]]
            p2 = self._densePoints[simplex[1]]
            p3 = self._densePoints[simplex[2]]
            # not have the save z (exclude triangles on the top and bottom surfaces)
            if not ((p1[2] == p2[2]) and (p1[2] == p3[2])):
                self._triangleList.append(Triangle(p1, p2, p3))
    
    
    def show_triangulated_surface(self, maps_color='tab20'):
        
        import matplotlib.pyplot as plt
        
        from matplotlib import colormaps
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        cmap = colormaps.get_cmap(maps_color)
        colors = [cmap(i%20) for i in range(len(self._triangleList))]
        
        for i, triangle in enumerate(self._triangleList):
            poly = Poly3DCollection([triangle.to_points()], alpha=0.6, facecolor=colors[i], edgecolor='k')
            ax.add_collection3d(poly)
        
        ax.scatter(self._densePoints[:,0], self._densePoints[:,1], self._densePoints[:,2], c='r', s=1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        plt.tight_layout()
        plt.show()



class Membrane:
    def __init__(self, membrane_model, pixel_size = 0.87):
        self.membrane_model = membrane_model
        self.__membrane_points = None
        self.__membrane_area = None
        self.__area_pixel_ratio = None
        self.pixel_size = pixel_size
        self.redo = True
        self.__membrane_trans = None
        self.__membrane_mean = None
        self.__boundary = None
        self.__convex_boundary = None
        self.__max_distance_to_boundary = None
        self.__membrane_vector = None


    @property
    def membrane_points(self):
        '''
        membrane in nm
        '''
        if self.__membrane_points is None:
            from subprocess import call
            import os
            pid = os.getpid()
            s="model2point {} {}_postmembrane.points > /dev/null".format(self.membrane_model, pid)
            # print(s)
            call(s,shell=True)
            self.__membrane_points=np.loadtxt('{}_postmembrane.points'.format(pid))

            s = "rm {}_postmembrane.points".format(pid)
            call(s,shell = True)
        return self.__membrane_points * self.pixel_size


    @property
    def membrane_area(self):
        '''
        Causion:Unit !!!!!!!!! Check next time u use it
        '''
        if self.__membrane_area is None:
            from subprocess import call,check_output
            if self.redo:
                s="imodmesh -l -T {}".format(self.membrane_model)
                call(s,shell=True)
            s="imodinfo {} | grep area | cut -d ' ' -f 6".format(self.membrane_model)
            area = check_output(s, shell=True)#call(s,shell=True)
            #self.__membrane_area = np.loadtxt('tmp') * self.pixel_size * self.pixel_size / 10000.0
            self.__membrane_area = float(area) * self.pixel_size * self.pixel_size / 10000.0
        return self.__membrane_area


    @property
    def area_pixel_ratio(self):

        if self.__area_pixel_ratio is None:
            self.__area_pixel_ratio = self.membrane_area / len(self.membrane_points)

        return self.__area_pixel_ratio


    def random_membrane_points(self, num_points = 100, remove_overlap = False, overlap_threshold = 7.0):
        #print(remove_overlap)
        if remove_overlap == False:
            rds=np.random.rand(num_points)
            idx=rds*len(self.membrane_points)
            idx=[int(item) for item in idx]
            return self.membrane_points[idx]

        else:
            result = []
            count = 0
            while count < num_points:
                one_point = self.random_membrane_points(num_points = 1, remove_overlap = False, overlap_threshold = overlap_threshold)[0]

                sign = 1
                for i,refPoint in enumerate(result):
                    if np.linalg.norm(one_point-refPoint) < overlap_threshold:
                        sign = 0
                        break

                if sign == 1:
                    result.append(one_point)
                    count = count + 1
            return np.array(result)

    def get_nearest_points(self, points):
        res=[]
        for item in points:
            a=np.linalg.norm(item-self.membrane_points,axis=1)
            res.append(self.membrane_points[np.argmin(a)])
        return np.array(res)


    @property
    def membrane_mean(self):
        if self.__membrane_mean is None:
            self.__membrane_mean = np.average(self.membrane_points, axis = 0)
        return self.__membrane_mean


    def ppp(self, membraneName):
        import numpy as np
        membrane = np.loadtxt(membraneName)
        N = membrane.shape[0]
        m = np.mean(membrane, axis = 0)
        data_m = membrane - np.tile(m, (N,1))
        covar = np.dot(data_m.T, data_m) / (N)
        U, S, V = np.linalg.svd(covar)
        V = V.T

        zVector = np.array([0,0,1])
        if np.abs(np.dot(zVector, V[:, 0])) + np.abs(np.dot(zVector, V[:, 1])) < 0.5:
            transVector = V[:, [0, 2]]
        else:
            transVector = V[:, :2]

        out = transVector
        np.savetxt(membraneName, out)


    @property
    def membrane_trans(self):
        if self.__membrane_trans is None:
            import os
            pid = os.getpid()
            np.savetxt('{}_membrane.point'.format(pid), self.membrane_points)
            
            from subprocess import call
            self.ppp('{}_membrane.point'.format(pid))
            membrane2D=np.loadtxt('{}_membrane.point'.format(pid))
            self.__membrane_trans=membrane2D[:3]

            s = "rm {}_membrane.point".format(pid)
            call(s,shell = True)

        return self.__membrane_trans


    def membrane_vector(self):
        '''
        Causion this vector can be either direction
        '''
        mat = self.membrane_trans
        if mat[0,0] * mat[1,1] - mat[0,1] * mat[1,0] == 0:
            z = 0
            if mat[0,0] != 0:
                y = 1
                x = mat[0,1]/mat[0,0]
            else:
                x = 1
                y = mat[0,0]/mat[0,1]
        else:
            z = 1
            a = np.array([mat[:2, 0],mat[:2, 1]])
            b = mat[2]
            (x,y)= np.linalg.solve(a, b)

        result = np.array([x,y,z])
        return result/ np.linalg.norm(result)


    @property
    def boundary(self):
        if self.__boundary is None:
            tmp = self.membrane_trans
        return self.__boundary


    @property
    def convex_boundary(self):
        if self.__convex_boundary is None:
            from scipy.spatial import ConvexHull
            membrane2D = self.transform2D(self.membrane_points)
            # membrane2D = np.reshape(self.transform2D(self.membrane_points), (-1,3))
            vertices = membrane2D[ConvexHull(membrane2D).vertices]
            from shapely.geometry import Polygon
            self.__convex_boundary=Polygon(vertices)
        return self.__convex_boundary


    def transform2D(self, point3D):
        return np.matmul(point3D - self.membrane_mean, self.membrane_trans)


    def distance_to_boundary(self, points, boundary = None):
        if boundary is None:
            boundary = self.boundary

        points = np.array(points)
        if len(points.shape) == 1:
            points = np.array([points])

        from shapely.geometry import Point
        def d(pt):
            pt_point = Point(pt)
            if boundary.contains(pt_point):
                return boundary.exterior.distance(pt_point)
            else:
                return -boundary.exterior.distance(pt_point)
        return np.array(list(map(d, points)))


    def max_membrane_distance_to_boundary(self, boundary = None):
        old = False
        if old:
            from scipy.spatial.distance import cdist
            b = np.array(list(self.boundary.exterior.coords))
            distance_matrix = cdist(self.transform2D(self.membrane_points), b)
            print(distance_matrix.shape)
            distance_min = np.min(distance_matrix, axis = 1)
            print(distance_min.shape)
            print(np.max(distance_min))
            return np.max(distance_min)
        else:
            if boundary is None:
                boundary = self.boundary

            from shapely.geometry import Point
            distance = 0
            for pt in self.transform2D(self.membrane_points):
                pt_point = Point(pt)
                if boundary.contains(pt_point):
                    d = boundary.exterior.distance(pt_point)
                    if d> distance:
                        distance = d
                else:
                    continue
            return distance


    @property
    def max_distance_to_boundary(self):
        if self.__max_distance_to_boundary is None:
            self.__max_distance_to_boundary = self.max_membrane_distance_to_boundary()
        return self.__max_distance_to_boundary



if __name__ == "__main__":
    a = VesicleList(0.87)
    a.fromXMLFile("pldel.xml")
