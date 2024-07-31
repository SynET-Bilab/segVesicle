import numpy as np
import SimpleITK as sitk

def resample_image(tomo, pixel_size=17.142, out_name=None, outspacing=17.142):
    tomo_sitk = sitk.ReadImage(tomo)
    # tomo_sitk = tomo_data
    #out_spacing = prepare_resample(tomo, pixel_size, outspacing)
    out_spacing = [outspacing, outspacing, outspacing]
    # original_spacing = tomo_sitk.GetSpacing()
    original_spacing = [pixel_size, pixel_size, pixel_size]
    original_size = tomo_sitk.GetSize()
    # if original_spacing[0] != 1:
    #     out_spacing = out_spacing
    # else:
    #     out_spacing = [0.942, 0.942, 0.942]
    out_spacing = [outspacing, outspacing, outspacing]
    out_size = [
        int(np.round(original_size[0] * original_spacing[0] / out_spacing[0])),
        int(np.round(original_size[1] * original_spacing[1] / out_spacing[1])),
        int(np.round(original_size[2] * original_spacing[2] / out_spacing[2]))
    ]
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(out_spacing)
    resample.SetSize(out_size)
    resample.SetOutputDirection(tomo_sitk.GetDirection())
    resample.SetOutputOrigin(tomo_sitk.GetOrigin())
    resample.SetTransform(sitk.Transform())
    resample.SetDefaultPixelValue(tomo_sitk.GetPixelIDValue())
    # resample.SetInterpolator(sitk.sitkBSpline)
    resample.SetInterpolator(sitk.sitkLinear)

    resample_tomo = resample.Execute(tomo_sitk)

    return resample_tomo