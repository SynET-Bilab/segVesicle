# Vesicle Segmentation

## Installation

1. Manage environment with conda

```shell
conda create --name=tomoSgmt python=3.6
```

2. Require tensorflow >= 2.0 

Please check your cuda and cudnn version and find a corresponding version for tensorflow

For example, you should install tensorflow with version 2.3.0 if you use cuda 10.1 

```shell
pip install tensorflow==2.3.0
```

3. Install other dependencies

```shell
pip install -r requirements.txt
```

4. Add environment variables

```shell
vi ~/.bashrc
export PATH=PATH_OF_THIS_FOLDER/bin/:$PATH
export PYTHONPATH=PARENT_PATH_OF_THIS_FOLDER/:$PYTHONPATH
```
