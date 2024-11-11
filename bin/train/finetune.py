import numpy as np
import mrcfile as mf
import os
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

import tensorflow.keras.backend as K
import time

from sklearn.metrics import precision_recall_fscore_support
import h5py
import matplotlib
matplotlib.use('agg')  # necessary else: AttributeError: 'NoneType' object has no attribute 'is_interactive'
import matplotlib.pyplot as plt
import sys
from datetime import datetime

# # 添加自定义模块路径
# sys.path.append('/share/data/CryoET_Data/lvzy/seg/seg_vesicles/')
from train import models
from train import losses

def train_model(dim_in, batch_size, epochs, load_model_pth, datadir, mode='iso'):
    """
    训练3D CNN模型进行分割任务。

    参数:
    - dim_in (int): 输入3D补丁的维度。
    - batch_size (int): 每批次的补丁数量。
    - epochs (int): 训练的轮数。
    - load_model_pth (str): 预训练模型的路径。如果为None，则不加载预训练模型。
    - datadir (str): 数据所在的目录。
    - mode (str): 数据模式，'iso' 或 'dec'。
    """
    
    # 设置其他参数
    Ncl = 2
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    loss = losses.tversky_loss
    flag_pretrained = True  # 是否加载预训练模型

    # 生成当前日期字符串
    current_date = datetime.now().strftime("%Y%m%d")
    # 创建结果文件夹路径，包含日期和模式
    result_folder = os.path.join(datadir, f"results_{current_date}_{mode}")
    
    # 根据 datadir 生成 input_txt 和 output_txt 的路径
    input_txt = os.path.join(datadir, "image.txt")
    output_txt = os.path.join(datadir, "label.txt")
    
    # 确保结果文件夹存在
    os.makedirs(result_folder, exist_ok=True)
    
    def random_erase_np(img, M):
        z_width = img.shape[0]
        y_width = img.shape[1]
        x_width = img.shape[2]

        for attempt in range(30):
            target_area = 15*15*15  # 15*15*15 的立方体区域
            w = int(np.round(np.cbrt(target_area)))
            if w < z_width and w < y_width and w < x_width:
                x1 = np.random.randint(0, x_width - w)
                y1 = np.random.randint(0, y_width - w)
                z1 = np.random.randint(0, z_width - w)

                img[z1:z1+w, y1:y1+w, x1:x1+w] = M
        return img

    def save_history(history, filename):
        if os.path.isfile(filename):  # 如果文件存在，删除后重新创建
            os.remove(filename)

        with h5py.File(filename, 'w') as h5file:
            # 保存训练和验证的loss
            h5file.create_dataset('loss', data=np.array(history['loss'], dtype='float16'))
            h5file.create_dataset('val_loss', data=np.array(history['val_loss'], dtype='float16'))

    def plot_history(history, filename):
        epochs = len(history['val_loss'])
        steps_per_valid = len(history['val_loss'][0]) if epochs > 0 else 1
        hist_loss_train = [np.mean(epoch_losses[-steps_per_valid:]) for epoch_losses in history['loss']]
        hist_loss_valid = [np.mean(epoch_losses) for epoch_losses in history['val_loss']]

        plt.figure(figsize=(5,6))
        plt.plot(hist_loss_train, label='train')
        plt.plot(hist_loss_valid, label='valid')
        plt.ylabel('loss')
        plt.xlabel('epochs')
        plt.legend()
        plt.grid()
        plt.savefig(filename)
        plt.close()

    def generate_batch_direct_read(datadir, batch_size, all_idx_list, all_data, all_label, mode, if_val=0):
        batch_data = np.zeros((batch_size, dim_in, dim_in, dim_in, 1))
        batch_target = np.zeros((batch_size, dim_in, dim_in, dim_in, Ncl))

        if len(all_idx_list) >= batch_size:
            index = random.sample(all_idx_list, batch_size)
        else:
            index = all_idx_list

        for i in range(len(index)):
            with mf.open(os.path.join(datadir, mode, all_data[index[i]])) as m:
                patch_data = m.data.astype(np.int8)
            with mf.open(os.path.join(datadir, 'label', all_label[index[i]])) as m:
                patch_target = m.data.astype(np.int8)
            if if_val == 0:
                if np.random.uniform() < 0.33:
                    M = np.mean(patch_data)
                    patch_data = random_erase_np(patch_data, M)
            # 归一化
            patch_data = (patch_data - np.mean(patch_data)) / np.std(patch_data)
            # one-hot编码
            patch_target_onehot = to_categorical(patch_target, Ncl)
            batch_data[i, :, :, :, 0] = patch_data
            batch_target[i] = patch_target_onehot
        return batch_data, batch_target

    def launch():
        # 初始化模型
        net = models.my_model(dim_in, Ncl)
        if flag_pretrained and load_model_pth and os.path.exists(load_model_pth):
            net = load_model(load_model_pth, custom_objects={'tversky_loss': loss})
        net.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        print('启动训练...')

        hist_loss_train = []
        hist_loss_valid = []
        process_time = []
        all_data = np.loadtxt(input_txt, dtype=str)
        all_label = np.loadtxt(output_txt, dtype=str)
        idx_list = list(range(len(all_data)))
        random.seed(1234)
        # 80% 训练，20% 验证
        idx_train_pool = random.sample(idx_list, int(0.8 * len(idx_list)))
        idx_val_pool = list(set(idx_list) - set(idx_train_pool))

        for e in range(epochs):
            # 训练
            start = time.time()
            list_loss_train = []
            steps_per_epoch_calc = len(idx_train_pool) // batch_size + 1
            for it in range(steps_per_epoch_calc):
                batch_data, batch_target = generate_batch_direct_read(datadir, batch_size, idx_train_pool, all_data, all_label, mode, if_val=0)
                loss_train = net.train_on_batch(batch_data, batch_target)
                print(f'epoch {e + 1}/{epochs} - it {it + 1}/{steps_per_epoch_calc} - loss: {loss_train[0]:.3f} - acc: {loss_train[1]:.3f}')
                list_loss_train.append(loss_train[0])

            hist_loss_train.append(list_loss_train)

            # 验证
            list_loss_valid = []
            steps_per_valid_calc = len(idx_val_pool) // batch_size + 1
            for it in range(steps_per_valid_calc):
                batch_data_valid, batch_target_valid = generate_batch_direct_read(datadir, batch_size, idx_val_pool, all_data, all_label, mode, if_val=1)
                loss_val = net.evaluate(batch_data_valid, batch_target_valid, verbose=0)
                list_loss_valid.append(loss_val[0])

            hist_loss_valid.append(list_loss_valid)
            end = time.time()
            process_time.append(end - start)
            print('-------------------------------------------------------------')
            print(f'EPOCH {e + 1}/{epochs} - valid loss: {loss_val[0]:.3f} - valid acc: {loss_val[1]:.3f} - {end - start:.2f}sec')

            # 保存和绘制训练历史
            history = {'loss': hist_loss_train, 'val_loss': hist_loss_valid}
            save_history(history, os.path.join(result_folder, 'net_train_history.h5'))
            plot_history(history, os.path.join(result_folder, 'net_train_history_plot.png'))
            print('=============================================================')
            
            # 每5个epoch保存一次模型权重
            if (e + 1) % 5 == 0:
                net.save(os.path.join(result_folder, f'net_weights_epoch{e + 1}.h5'))
        
        print(f"模型训练总共耗时 {np.sum(process_time):.2f} 秒")
        net.save(os.path.join(result_folder, 'net_weights_FINAL.h5'))
        print(f"最终模型已保存到 {os.path.join(result_folder, 'net_weights_FINAL.h5')}")
    
    # 启动训练
    launch()

#示例调用：
# train_model(
#     dim_in=64,
#     batch_size=20,
#     epochs=100,
#     load_model_pth='/home/liushuo/Documents/code/vesiclePipeline/segVesicle/pretrained/vesicle_seg_model_2.h5',
#     datadir='/home/liushuo/Documents/data/stack-out_demo/subtomo/',
#     mode='iso'  # 或 'dec'
# )
