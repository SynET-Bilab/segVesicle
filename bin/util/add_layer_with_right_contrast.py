import napari.viewer
import numpy as np
import napari

def add_layer_with_right_contrast(data: np, name: str, viewer: napari.Viewer):
    """
    将tomo添加到viewer，并设置图层属性。

    参数:
    tomo: numpy数组，表示体数据。
    name: str，图层的名称。
    viewer: napari viewer对象。
    """
    # 计算百分位数
    min_val = np.percentile(data, 0.1)  # 计算第0.1百分位数
    max_val = np.percentile(data, 99)   # 计算第99百分位数
    
    # 添加图像到viewer并设置属性
    viewer.add_image(data, name=name)
    viewer.layers[name].contrast_limits = [min_val, max_val]
    viewer.layers[name].opacity = 0.8
    viewer.layers[name].gamma = 0.7