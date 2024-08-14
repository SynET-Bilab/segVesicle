import mrcfile

from scipy.ndimage import zoom

def get_tomo(input_data):
    with mrcfile.open(input_data) as m:
        data = m.data
    return data

def get_tomo(input_data):
    with mrcfile.open(input_data) as m:
        data = m.data
    return data


def resample_image(tomo, pixel_size, out_name=None, outspacing=17.142):

    original_data = get_tomo(tomo)
    
    original_spacing = [pixel_size, pixel_size, pixel_size]
    out_spacing = [outspacing, outspacing, outspacing]
    scale_factors = [
        original_spacing[0] / out_spacing[0],
        original_spacing[1] / out_spacing[1],
        original_spacing[2] / out_spacing[2]
    ]
    # resample
    out_data = zoom(original_data, zoom=scale_factors, order=1)

    return out_data