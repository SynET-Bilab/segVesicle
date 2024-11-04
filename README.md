# segVesicle

DL-based vesicle segmentation program for cryo electron tomography images, with user-friendly GUI based on [napari](https://napari.org/stable/) for manual correction.

<p align='center'>
<video src="https://github.com/SynET-Bilab/segVesicle/assets/81632333/28faff9e-360e-4aa1-8fa6-db1d87d6ba4c" controls="controls" width="800px">
</p>

## Installation

1. Source code installation. Both `segVesicle` and `IsoNet` are needed. [IsoNet](https://github.com/IsoNet-cryoET/IsoNet) is used for missing wedge restoration and denoising to improve the segmentation performance. 

```bash
git clone git@github.com:IsoNet-cryoET/IsoNet.git
git clone git@github.com:SynET-Bilab/segVesicle.git
```

2. Install basic dependencies

```bash
conda env create -f environment.yml
conda activate segVesicle
```

3. `segVesicle` relies on tensorflow-gpu2, the specific version of `tensorflow-gpu` should be determined by the version of `cuda` . For the correspondence, see https://www.tensorflow.org/install/source?hl=en#gpu. For example, if you use `cuda-11.2`, `tensorflow-gpu` 2.5.0 to 2.11.0 is ok. `cudnn` with proper version should also be installed. 

```bash
pip install tensorflow-gpu==2.x.x
```

4. Add environment variables

```bash
export PATH=PATH_TO_ISONET_FOLDER/bin:$PATH 
export PYTHONPATH=PATH_TO_PARENT_FOLDER_OF_ISONET_FOLDER:$PYTHONPATH

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