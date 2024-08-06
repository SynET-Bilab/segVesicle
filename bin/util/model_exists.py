import os
import requests

def ensure_model_exists(model_name):
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pretrained_path = os.path.join(base_path, 'pretrained')
    
    # Model names and their corresponding download links
    models = {
        'vesicle_seg_model_1.h5': 'https://recstore.ustc.edu.cn/file/20240705_63842bdaab37cc8d531cda82be501c1a?Signature=l6+iNWukNEpk+i2cMP2I1vak+Po=&Expires=1723017349&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22vesicle_corrected_model.h5%22&storage=moss&filename=vesicle_corrected_model.h5&download=download',
        'vesicle_seg_model_2.h5': 'https://recstore.ustc.edu.cn/file/20240705_2c371fe0a24553e66d93ed839c492a92?Signature=Obzd0ZmaboLVkyrkKX+jCaDb3HE=&Expires=1723017288&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22vesicle_seg_model_2.h5%22&storage=moss&filename=vesicle_seg_model_2.h5&download=download',
        'vesicle_corrected_model.h5': 'https://recstore.ustc.edu.cn/file/20240705_63842bdaab37cc8d531cda82be501c1a?Signature=6Md/cNOdCaE/7Ao92PKroSIMm7M=&Expires=1723017000&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22vesicle_corrected_model.h5%22&storage=moss&filename=vesicle_corrected_model.h5&download=download'
    }
    
    # Check if the input model name is in the dictionary
    if model_name not in models:
        print(f"The model name {model_name} is not in the known models list.")
        return None
    
    model_filepath = os.path.join(pretrained_path, model_name)
    
    # Check if the pretrained folder exists, if not, create it
    if not os.path.exists(pretrained_path):
        os.makedirs(pretrained_path)
    
    # Check if the model file exists
    if not os.path.isfile(model_filepath):
        print(f"{model_name} does not exist, downloading from the remote server...")
        
        # Download the model file
        url = models[model_name]
        response = requests.get(url, stream=True)
        
        # Check if the request was successful
        if response.status_code == 200:
            with open(model_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{model_name} download completed, saved at: {model_filepath}")
        else:
            print(f"Download failed, status code: {response.status_code}")
            return None
    else:
        print(f"{model_name} already exists, path: {model_filepath}")

    return model_filepath
