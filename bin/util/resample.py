import SimpleITK as sitk
import numpy as np
from scipy.ndimage import zoom


def resample_image(tomo, pixel_size, outspacing=17.142):
    # 使用 SimpleITK 读取图像
    tomo_sitk = sitk.ReadImage(tomo)

    original_spacing = [pixel_size, pixel_size, pixel_size]
    out_spacing = [outspacing, outspacing, outspacing]
    
    # 获取原始图像数据
    original_data = sitk.GetArrayFromImage(tomo_sitk)
    
    # 计算缩放因子
    scale_factors = [
        original_spacing[0] / out_spacing[0],
        original_spacing[1] / out_spacing[1],
        original_spacing[2] / out_spacing[2]
    ]
    
    # 使用 scipy.ndimage.zoom 进行重采样
    out_data = zoom(original_data, zoom=scale_factors, order=1)
    

    return out_data