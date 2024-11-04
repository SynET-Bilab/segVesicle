#!/usr/bin/env python
# encoding: utf-8
import os
import numpy as np
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from tqdm import tqdm
from lxml import etree
import tempfile
import subprocess
import pandas as pd


# save_points_as_mod(points, object_id=1, model_file="/home/liushuo/Documents/data/stack-out_demo/p2/ves_seg/vesicle_analysis/sampled_points.mod")

# def save_points_as_mod(points: np.ndarray, object_id: int, model_file: str):
#     """
#     Saves the sampled points as a .mod file, grouping them by z and creating contours.
    
#     :param points: An array of points to save, where each unique z defines a new group.
#     :param object_id: The object ID to use in the .mod file.
#     :param model_file: The path to save the .mod file.
#     """
#     # Round z values to the nearest integer
#     points[:, 2] = np.round(points[:, 2]).astype(int)
    
#     # Group points by z-value
#     z_groups = {}
#     for point in points:
#         z = point[2]
#         if z not in z_groups:
#             z_groups[z] = []
#         z_groups[z].append(point)
    
#     # Prepare data for DataFrame
#     data = []
#     contour_count = 0
#     for z, group in z_groups.items():
#         if len(group) > 1:  # Check group size and limit contour count
#             for point in group:
#                 # object_id, contour_id, x, y, z (1-based for object and contour)
#                 data.append([object_id, contour_count + 1, point[0], point[1], point[2]])
#             contour_count += 1

#     # Create DataFrame and write to .mod file
#     df = pd.DataFrame(data, columns=["object", "contour", "x", "y", "z"])
#     write_model(model_file, df)

# def write_model(model_file: str, model_df: pd.DataFrame):
#     """
#     Converts the point data to a .mod file format using the IMOD tool point2model.
    
#     :param model_file: Path where the .mod file will be saved.
#     :param model_df: DataFrame containing the point data.
#     """
#     model = np.asarray(model_df)

#     # Ensure the directory exists
#     model_dir = os.path.dirname(model_file)
#     if not os.path.exists(model_dir):
#         os.makedirs(model_dir)

#     # Save points to a temporary file and convert to .mod
#     with tempfile.NamedTemporaryFile(suffix=".pt", dir=".") as temp_file:
#         # Save point data to a temporary .pt file
#         point_file = temp_file.name
#         np.savetxt(point_file, model, fmt=(['%d']*2 + ['%.2f']*3))

#         # Use point2model to convert the point file to a .mod file
#         cmd = f"point2model -op {point_file} {model_file} >/dev/null"
#         subprocess.run(cmd, shell=True, check=True)
#     print(f".mod file saved successfully to {model_file}")



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

        classElement = vesicleElement.xpath('Class')[0]
        self._radius = float(classElement.get('Name')) / pixelSize /2.0


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
                   "Type"]
        argStrList = ["Type"]
        
        for item in argList:
            itemPath = vesicleElement.xpath(item)

            if len(itemPath) > 0:
                its = itemPath[0].items()

                if item in argStrList:
                    setattr(self, f"_{item[0].lower()}{item[1:]}", its[0][1])
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
            for i, evec in enumerate(self._evecs):
                vesicleElement.append(etree.Element("Evecs",\
                                                    X = str(evec[0]),\
                                                    Y = str(evec[1]),\
                                                    Z = str(evec[2]),\
                                                    idx = str(i)))

        if hasattr(self,"_distance"):
            vesicleElement.append(etree.Element("Distance",\
                                                d = str(self._distance)))

        if hasattr(self,"_projectionPoint"):
            vesicleElement.append(etree.Element("ProjectionPoint",\
                                                X = str(self._projectionPoint[0]),\
                                                Y = str(self._projectionPoint[1]),\
                                                Z = str(self._projectionPoint[2])))

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
        points = np.stack([x, y, z], axis=-1)  # (num_phi, num_theta, 3)
        
        points *= self._radius3D
        points = points @ self._evecs + self._center3D
        
        # assert points.shape == (precision, 3), f"Unexpected shape: {points.shape}"
        
        return points
    
    
    def sample_on_vesicle_3d(self, precision : int) -> np.ndarray:
        '''
        to get random points on the surface of a 3D ellipsoid
        
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
        return self._radius

    def getRadius2D(self):
        return self._radius2D
    
    def getRadius3D(self):
        return self._radius3D

    def setRadius(self, radius):
        self._radius = radius
    
    def setRadius2D(self, radius2D):
        self._radius2D = radius2D
    
    def setRadius3D(self, radius3D):
        self._radius3D = radius3D
    
    def setEvecs(self, evecs):
        self._evecs = evecs
    
    # ls
    def setRotation2D(self, rotation2D):
        self._rotation2D = rotation2D
    
    def getEvecs(self):
        return self._evecs

    def getId(self):
        return self._vesicleId

    def getPixelSize(self):
        return self._pixelSize

    def getProjectionPoint(self):
        return self._projectionPoint

    def getDistance(self):
        return self._distance

    def setDistance(self,d):
        self._distance = d

    def getType(self):
        return self._type
    
    def setType(self,t):
        self._type = t


class VesicleList:
    """parameters(radius, distance, position etc.) stores in pixel. If you want to use nm, please multiply self.getPixelSize
    """
    def __init__(self, pixelSize = 1.0):
        self._vesicleList = []
        self._pixelSize = pixelSize

    def __len__(self):
        return len(self._vesicleList)

    def __setitem__(self, key, value):
        self._vesicleList[key]=value

    def __getitem__(self, key):
        """retrive vesicle at position defined by key
        """

        if isinstance(key, int):
            if key < len(self):
                return self._vesicleList[key]
            else:
                raise IndexError('Index out of range')

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
        for i,vesicle in tqdm(enumerate(self._vesicleList)):
            # for pits defined by three points, set distance to 0 and projection point is the center
            if vesicle.getType() == 'pit':
                vesicle._distance = 0.
                vesicle._projectionPoint = vesicle.getCenter()
                self._distance.append(0)
                self._projectionPoint.append(vesicle.getCenter())
                nearest_point_list.append(vesicle.getCenter())
                modtxtFile.append(np.concatenate((np.array([1, i+1]), vesicle.getCenter())))
                modtxtFile.append(np.concatenate((np.array([1, i+1]), vesicle.getCenter())))

            elif vesicle.getType() == 'vesicle':
                dis, PP0, nearest_point = vesicle.distance_to_surface(surface, precision, tree, surface._densePoints)
                self._distance.append(dis)
                self._projectionPoint.append(PP0)
                nearest_point_list.append(nearest_point)
                modtxtFile.append(np.concatenate((np.array([1, i+1]), nearest_point)))
                modtxtFile.append(np.concatenate((np.array([1, i+1]), PP0)))
        
        modtxtFile = np.asarray(modtxtFile)
        np.savetxt('nearest_point.txt', np.reshape(modtxtFile, (-1, 5)), fmt='%d')
        cmd = 'point2model -sp 10 nearest_point.txt nearest_point.mod'
        os.system(cmd)
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


    def point_triangle_distance(self, p, standard = False):
        """calculate distance from a point to the triangle.
        @param p: a list of 3 floats
        @param standard: use the standard way to measure distance.
        if not, use sampling points in the triangle to estamate the distance
        if not, requires the triangle be sampled by self.sampling_triangle()
        @return distance, nearest point:
        """
        if standard:
            return self.point_triangle_distance_standard(p)
        else:
            return self.point_triangle_distance_using_sample_points(p)


    def point_triangle_distance_standard(self, p):
        """calculate distance from a point to the triangle.

        """
        import numpy as np
        p = np.array(p, dtype = np.float)
        from scipy.io import savemat
        savemat('values.mat', {'triangle':[self.p1,self.p2,self.p3],'p':p})
        from subprocess import check_output
        s = "matlab -nodisplay  -r \"load('values.mat');[dis, PP0]=pointTriangleDistance(triangle,p);disp(num2str(dis));disp(num2str(PP0)); quit;\" | sed -n '11,12p'"
        distance =  check_output(s, shell=True)
        distance = distance.split()
        distance = list(map(float,distance))
        return distance[0], distance[1:4]


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


    # def area(self):
    #     """
    #     Calculate area of a given triangle
    #     """
    #     import numpy as np
    #     a = np.linalg.norm(self.p3 - self.p2)
    #     b = np.linalg.norm(self.p1 - self.p3)
    #     c = np.linalg.norm(self.p2 - self.p1)
    #     p = (a + b + c) / 2
    #     return (abs(p*(p - a)*(p - b)*(p - c)))**0.5
    
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
        output: a np.ndarray with shape (3, 3)
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
    
    
    def from_model_auto_segment(self, model, objNum=2):
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
        
        def filter_equ_x(points, idx):
            '''
            get average y of points with equal x
            '''
            obj, _, x, _, z = points[0]
            y = custom_round(points[:, 3].mean()).astype(np.float64)
            return np.array([obj, idx, x, y, z])
        
        def max_filter(unfiltered):
            '''
            to fix conflicts that in the same contour, points with the same x have different y (here just do a adjusted NMS by mean y)
            '''
            filtered = []
            contours = []
            
            for z in sorted(list(set(unfiltered[:, -1].tolist()))):
                contours.append(unfiltered[unfiltered[:, -1] == z])
            
            for i, contour in enumerate(contours):
                idx = i + 1
                point_x_set = sorted(list(set(contour[:, 2].tolist())))
                for x in point_x_set:
                    point_equ_x = contour[contour[:, 2] == x]
                    filtered.append(filter_equ_x(point_equ_x, idx))
            filtered = np.array(filtered)
            
            return filtered
        
        cmd = 'model2point -ob {} {} >> /dev/null'.format(model, model.replace('.mod', '.point'))
        os.system(cmd)
        untreated = np.loadtxt(model.replace('.mod', '.point'))
        untreated = untreated[untreated[:, 0] == objNum]
        
        membrane = max_filter(untreated)
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




if __name__ == "__main__":
    a = VesicleList(0.87)
    a.fromXMLFile("pldel.xml")
