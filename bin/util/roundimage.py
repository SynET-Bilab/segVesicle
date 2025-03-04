# import cv2
# import numpy as np

# def draw_red_ellipse(gray_image, center, a, b, e1):
#     """
#     在灰度图像上绘制红色椭圆并保存
    
#     参数:
#     gray_image (numpy.ndarray): 输入的灰度图像（二维数组）
#     center (tuple): 椭圆中心坐标 (x, y)
#     a (int): 长轴的半长
#     b (int): 短轴的半长
#     e1 (list/np.array): 长轴方向向量 [ex, ey]
#     """
#     # 将灰度图转换为三通道图像
#     color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
#     # 计算旋转角度
#     e1 = np.array(e1, dtype=np.float32)
#     norm = np.linalg.norm(e1)
#     if np.isclose(norm, 0):
#         raise ValueError("方向向量e1不能是零向量")
#     e1_normalized = e1 / norm
#     theta_deg = np.degrees(np.arctan2(e1_normalized[1], e1_normalized[0]))
    
#     # 绘制椭圆
#     cv2.ellipse(
#         img=color_image,
#         center=center,
#         axes=(a, b),
#         angle=theta_deg,
#         startAngle=0,
#         endAngle=360,
#         color=(0, 0, 255),  # 红色（BGR格式）
#         thickness=2
#     )
    
#     # 保存图像
#     cv2.imwrite('output_image_with_ellipse.png', color_image)
#     return color_image

# # 示例用法
# if __name__ == "__main__":
#     import json
#     import mrcfile
#     # 创建示例灰度图像（512x512）
#     jsonfile = '/share/data/CryoET_Data/synapse/synapse202501/20250126_20250123-20250107-g2b4_40ms24C/stack-out/pp120-1/ves_seg/pp120_vesicle.json'
#     tomofile  = '/share/data/CryoET_Data/synapse/synapse202501/20250126_20250123-20250107-g2b4_40ms24C/stack-out/pp120-1/ves_seg/pp120_wbp_corrected.mrc'
#     margin = 15
#     with open(jsonfile) as f:
#         ves = json.load(f)
#     vesicle_info = ves['vesicles']
#     with mrcfile.open(tomofile) as m:
#         mrcdata = m.data

#     for info in (vesicle_info[0],):
#         center = np.array(info['center2D'])
#         radius = np.array(info['radius2D'])
#         r_ma = np.max(radius).astype(np.int8)
#         cenz = np.round(center[0]).astype(np.int16)
#         ceny = np.round(center[1]).astype(np.int16)
#         cenx = np.round(center[2]).astype(np.int16)
#         gray_img = mrcdata[cenz,ceny-r_ma-margin:ceny+r_ma+margin+1,cenx-r_ma-margin:cenx+r_ma+margin+1] + 128 #int8 to uint8
#         gray_img = gray_img.astype(np.uint8)
#         center_point = (r_ma+margin, r_ma+margin)
#         if radius[0] < radius[1]:
#             a_length = radius[1] # 长轴半长
#             b_length = radius[0] # 短轴半长
#             direction_vector = info['evecs2D'][1]  # 长轴方向向量
#         else:
#             a_length = radius[0] # 长轴半长
#             b_length = radius[1] # 短轴半长
#             direction_vector = info['evecs2D'][0]  # 长轴方向向量
#         print(gray_img.shape,center_point,direction_vector)
#         result_image = draw_red_ellipse(gray_img, center_point, a_length, b_length, direction_vector)


import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib

def draw_precision_ellipse(outputpath, gray_image, center, a, b, e1, dpi=300):
    """
    使用Matplotlib绘制高精度椭圆
    
    参数:
    gray_image (numpy.ndarray): 输入的灰度图像（二维数组）
    center (tuple): 椭圆中心坐标 (x, y)
    a (float): 长轴的半长
    b (float): 短轴的半长
    e1 (list/np.array): 长轴方向向量 [ex, ey]
    dpi (int): 输出图像分辨率
    
    返回:
    numpy.ndarray: 带有椭圆的RGB图像
    """
    save_dir = 'roundImages_vesseg'
    if not os.path.exists('{}/{}'.format('vesicle_analysis', save_dir)):
        s = "mkdir {}/{}".format('vesicle_analysis', save_dir)
        os.system(s)
    # 创建Matplotlib画布
    fig = plt.figure(figsize=(gray_image.shape[1]/dpi, gray_image.shape[0]/dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], aspect='auto')
    
    # 显示原始图像（Matplotlib使用RGB格式）
    ax.imshow(gray_image, cmap='gray', origin='upper')

    # 计算旋转角度（单位：度）
    e1 = np.array(e1)
    norm = np.linalg.norm(e1)
    if np.isclose(norm, 0):
        plt.close()
        raise ValueError("方向向量e1不能是零向量")
    angle = np.degrees(np.arctan2(e1[1], e1[0]))

    # 创建椭圆对象（注意：width和height是全长直径）
    ellipse = Ellipse(
        xy=center,
        width=2*a,      # 总长轴长度
        height=2*b,     # 总短轴长度
        angle=angle,
        edgecolor='red',
        facecolor='none',
        linewidth=0.2,
        antialiased=True
    )

    # 添加椭圆到画布
    ax.add_patch(ellipse)
    
    # 关闭坐标轴和边距
    ax.set_axis_off()
    
    # 渲染图像
    fig.canvas.draw()
    
    # 转换为numpy数组
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close()
    
    # 保存图像
    matplotlib.image.imsave(outputpath, img_array)
    return img_array

# 示例用法
if __name__ == "__main__":
    import json
    import mrcfile
    # 创建示例灰度图像（512x512）
    jsonfile = 'pp472_vesicle.json'
    tomofile  = 'pp472_wbp_corrected.mrc'
    margin = 15
    with open(jsonfile) as f:
        ves = json.load(f)
    vesicle_info = ves['vesicles']
    with mrcfile.open(tomofile) as m:
        mrcdata = m.data
    maxvalue=np.max(mrcdata)
    mrcdatapad = np.pad(mrcdata,20,'constant',constant_values= maxvalue)
    # for i,info in enumerate((vesicle_info[55],)):
    for i,info in enumerate(vesicle_info):
        center = np.array(info['center2D']) + 20
        radius = np.array(info['radius2D'])
        r_ma = np.max(radius).astype(np.int8)
        cenz = np.round(center[0]).astype(np.int16)
        ceny = np.round(center[1]).astype(np.int16)
        cenx = np.round(center[2]).astype(np.int16)
        gray_img = mrcdatapad[cenz,ceny-r_ma-margin:ceny+r_ma+margin+1,cenx-r_ma-margin:cenx+r_ma+margin+1] + 128 #int8 to uint8
        gray_img = gray_img.astype(np.uint8)
        center_point = (r_ma+margin, r_ma+margin)
        if radius[0] < radius[1]:
            a_length = radius[1] # 长轴半长
            b_length = radius[0] # 短轴半长
            direction_vector = [info['evecs2D'][1][1],info['evecs2D'][1][0]]  # 长轴方向向量
        else:
            a_length = radius[0] # 长轴半长
            b_length = radius[1] # 短轴半长
            direction_vector = [info['evecs2D'][0][1],info['evecs2D'][0][0]]  # 长轴方向向量

    
        # 调用函数绘制椭圆
        outputpath = f'vesicle_analysis/roundImages_vesseg/{i}.png'
        print(center, a_length, b_length, direction_vector)
        esult_img = draw_precision_ellipse(outputpath, gray_img, center_point, a_length, b_length, direction_vector)