import os
import requests

def ensure_model_exists(model_name):
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    pretrained_path = os.path.join(base_path, 'pretrained')
    
    # 文件名和对应的下载链接
    models = {
        'vesicle_seg_model_1.h5': 'https://recstore.ustc.edu.cn/file/20240705_63842bdaab37cc8d531cda82be501c1a?Signature=l6+iNWukNEpk+i2cMP2I1vak+Po=&Expires=1723017349&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22vesicle_corrected_model.h5%22&storage=moss&filename=vesicle_corrected_model.h5&download=download',
        'vesicle_seg_model_2.h5': 'https://recstore.ustc.edu.cn/file/20240705_2c371fe0a24553e66d93ed839c492a92?Signature=Obzd0ZmaboLVkyrkKX+jCaDb3HE=&Expires=1723017288&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22vesicle_seg_model_2.h5%22&storage=moss&filename=vesicle_seg_model_2.h5&download=download',
        'vesicle_corrected_model.h5': 'https://recstore.ustc.edu.cn/file/20240705_63842bdaab37cc8d531cda82be501c1a?Signature=6Md/cNOdCaE/7Ao92PKroSIMm7M=&Expires=1723017000&AccessKeyId=MAKIG23JM2UB98N0KTQH&response-content-type=application%2Foctet-stream&response-content-disposition=attachment%3Bfilename%3D%22vesicle_corrected_model.h5%22&storage=moss&filename=vesicle_corrected_model.h5&download=download'
    }
    
    # 检查输入的模型名称是否在字典中
    if model_name not in models:
        print(f"模型名称 {model_name} 不在已知模型列表中。")
        return None
    
    model_filepath = os.path.join(pretrained_path, model_name)
    
    # 检查 pretrained 文件夹是否存在，如果不存在则创建它
    if not os.path.exists(pretrained_path):
        os.makedirs(pretrained_path)
    
    # 检查模型文件是否存在
    if not os.path.isfile(model_filepath):
        print(f"{model_name} 不存在，正在从远程服务器下载...")
        
        # 下载模型文件
        url = models[model_name]
        response = requests.get(url, stream=True)
        
        # 检查请求是否成功
        if response.status_code == 200:
            with open(model_filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"{model_name} 下载完成，保存路径：{model_filepath}")
        else:
            print(f"下载失败，状态码：{response.status_code}")
            return None
    else:
        print(f"{model_name} 已经存在，路径：{model_filepath}")

    return model_filepath