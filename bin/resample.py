import SimpleITK as sitk
import numpy as np
import mrcfile
import fire
from scipy.ndimage import zoom

def generate_new_tomo(out_data, out_name, outspacing):
    with mrcfile.new(out_name, overwrite=True) as m:
        m.set_data(out_data)
        m.voxel_size = [outspacing, outspacing, outspacing]

def resample_image(tomo, pixel_size, out_name=None, outspacing=17.142):
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
    
    if out_name is None:
        out_name = tomo[:-4] + '-resample' + tomo[-4:]
    
    generate_new_tomo(out_data, out_name, outspacing)

    return out_data


def measure(tomo, pixel_size, outspacing=17.142):
    sitk_tomo = sitk.ReadImage(tomo)
    # original_spacing = sitk_tomo.GetSpacing()
    original_spacing = [pixel_size, pixel_size, pixel_size]
    # if original_spacing[0] != 1:
    #     original_spacing = original_spacing
    # else:
    #     original_spacing = [pixel_size, pixel_size, pixel_size]
    original_spacing = [pixel_size, pixel_size, pixel_size]
    original_size = sitk_tomo.GetSize()
    #output_spacing = prepare_resample(tomo, pixel_size, outspacing)
    output_spacing = [outspacing, outspacing, outspacing]
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / output_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / output_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / output_spacing[2]))
    ]
    #out_spacing = [outspacing, outspacing, outspacing]
    return [original_spacing, original_size, output_spacing, out_size]


def main(tomo, pixel_size, outname, outspacing=17.142):
    resample_image(tomo, pixel_size, outname, outspacing=17.142)


if __name__ == "__main__":
    fire.Fire(main)
    # main('/home/liushuo/Documents/data/stack-out_demo/p187/p187-bin4-15i.rec', 4, '/home/liushuo/Documents/data/stack-out_demo/p187/p187-4.rec')