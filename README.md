# Vesicle3D

DL-based vesicle segmentation program for cryo electron tomography images, with user-friendly GUI based on [napari](https://napari.org/stable/) for manual correction.

<p align='center'>
<video src="https://github.com/SynET-Bilab/segVesicle/assets/81632333/28faff9e-360e-4aa1-8fa6-db1d87d6ba4c" controls="controls" width="800px">
</p>

## Installation

1. Source code installation. Both `Vesicle3D` and `IsoNet` are needed. [IsoNet](https://github.com/IsoNet-cryoET/IsoNet) is used for missing wedge restoration and denoising to improve the segmentation performance. 

```bash
git clone git@github.com:IsoNet-cryoET/IsoNet.git
git clone git@github.com:SynET-Bilab/segVesicle.git
```

2. Install basic dependencies

```bash
conda env create -f environment.yml
conda activate segVesicle
```

3. Proper version of `cuda` and `cudnn` should be installed. For specific correspondence, see https://www.tensorflow.org/install/source?hl=en#gpu.


4. Add environment variables

```bash
# IsoNet
export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 
export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH
# Vesicle3D
export PATH=PATH_TO_SEGVESICLE_FOLDER/bin:$PATH
export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_SEGVESICLE_FOLDER:$PYTHONPATH
```

5. Open GUI

```bash
cd PARENT_PATH_OF_TOMO/
segvesicle.py
```

## Tutorial

For for details, please refer to GUI

![Tutorial](bin/help_files/img/tutorial.png)