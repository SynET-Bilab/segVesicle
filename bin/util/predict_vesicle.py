try:
    from util.model_exists import ensure_model_exists
except ImportError:
    from segVesicle.bin.util.model_exists import ensure_model_exists

try:
    from segVesicle.bin.morph import (
        morph_process,
        vesicle_measure,
        vesicle_rendering,
    )
except ImportError:
    from morph import (
        morph_process,
        vesicle_measure,
        vesicle_rendering,
    )



def predict_label(deconv_data, corrected_data):
    try:
        from segVesicle.bin.segment import combine_segmentations, segment_array
    except ImportError:
        from segment import combine_segmentations, segment_array

    model_1 = 'vesicle_seg_model_1.h5'
    model_2 = 'vesicle_seg_model_2.h5'
    
    path_weights1 = ensure_model_exists(model_1)
    path_weights2 = ensure_model_exists(model_2)

    patch_size = 192
    pcrop_corrected = max((patch_size - corrected_data.shape[0])//2 + 2 ,48)
    pcrop_deconv = max((patch_size - deconv_data.shape[0])//2 + 2 ,48)
    seg1 = segment_array(
        path_weights1,
        corrected_data,
        patch_size=patch_size,
        pcrop=pcrop_corrected,
        pad_mode="constant",
        constant_values=0,
    )
    seg2 = segment_array(
        path_weights2,
        deconv_data,
        patch_size=patch_size,
        pcrop=pcrop_deconv,
        pad_mode="constant",
        constant_values=0,
    )
    labelmap = combine_segmentations(seg1, seg2)
    
    return labelmap
