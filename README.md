# Vesicle Segmentation

## Installation
1. Require tensorflow >= 2.0 

Please check your cuda and cudnn version and find a corresponding version for tensorflow

For example, you should install tensorflow with version 2.3.0 if you use cuda 10.1 
```
pip install tensorflow==2.3.0
```

2. Install other dependencies
   
```
pip install -r requirements.txt
```

3. Add environment variables

```
vi ~/.bashrc
export PATH=PATH_OF_THIS_FOLDER/bin/:$PATH
export PYTHONPATH=PARENT_PATH_OF_THIS_FOLDER/:$PYTHONPATH
```
