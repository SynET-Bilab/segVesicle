'''

'''
import mrcfile
import numpy as np
#from skimage.morphology import opening,closing, disk
from tensorflow.keras.utils import Sequence


class DataWrapper(Sequence):

    def __init__(self, X,  batch_size):
        self.X = X
        self.batch_size = batch_size
    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))
    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        return self.X[idx]

def normalize(x, percentile = True, pmin=4.0, pmax=96.0, axis=None, clip=False, eps=1e-20):
    """Percentile-based image normalization."""

    if percentile:
        mi = np.percentile(x,pmin,axis=axis,keepdims=True)
        ma = np.percentile(x,pmax,axis=axis,keepdims=True)
        out = (x - mi) / ( ma - mi + eps )
        out = out.astype(np.float32)
        if clip:
            return np.clip(out,0,1)
        else:
            return out
    else:
        out = (x-np.mean(x))/np.std(x)
        out = out.astype(np.float32)
        return out


def toUint8(data):
    data=np.real(data)
    data=data.astype(np.double)
    ma=np.max(data)
    mi=np.min(data)
    data=(data-mi)/(ma-mi)*255
    data=np.clip(data,0,255)
    data=data.astype(np.uint8)
    return data


def gene_2d_training_data(tomo,mask,sample_mask=None,num=100,sidelen=128,neighbor_in=5, neighbor_out=1):
    with mrcfile.open(tomo) as o:
        orig_tomo=o.data 
    tomo = normalize(orig_tomo,percentile = False)
    tomo = toUint8(tomo)
    
    with mrcfile.open(mask) as m:
        mask=m.data

    if  sample_mask != None:
        with mrcfile.open(sample_mask) as sm:
            sample_mask = sm.data
    else:
        sample_mask = np.ones(tomo.shape)
    sp=tomo.shape
    if sample_mask is None:
        sample_mask=np.ones(sp)
    else:
        sample_mask=sample_mask
    
    border_slices = tuple([slice(s // 2, d - s + s // 2 + 1) for s, d in zip((neighbor_in,sidelen,sidelen), sp)])
    valid_inds = np.where(sample_mask[border_slices])
    valid_inds = [v + s.start for s, v in zip(border_slices, valid_inds)]
    sample_inds1 = np.random.choice(len(valid_inds[0]), num, replace=len(valid_inds[0]) < num)
    sample_inds2 = np.random.choice(len(valid_inds[0]), int(num*0.1), replace=len(valid_inds[0]) < int(num*0.1))
    rand_inds1 = [v[sample_inds1] for v in valid_inds]
    rand_inds2 = [v[sample_inds2] for v in valid_inds]
    seeds1 = (rand_inds1[0],rand_inds1[1], rand_inds1[2])
    seeds2 = (rand_inds2[0],rand_inds2[1], rand_inds2[2])
    

    X_train = np.swapaxes(crop_patches(tomo,seeds1,sidelen=sidelen,neighbor=neighbor_in),1,-1)
    Y_train = np.swapaxes(crop_patches(mask,seeds1,sidelen=sidelen,neighbor=neighbor_out),1,-1)
    X_test = np.swapaxes(crop_patches(tomo,seeds2,sidelen=sidelen,neighbor=neighbor_in),1,-1)
    Y_test = np.swapaxes(crop_patches(mask,seeds2,sidelen=sidelen,neighbor=neighbor_out),1,-1)

    print(X_train.shape)
    return (X_train,Y_train), (X_test,Y_test)

def crop_patches(img3D,seeds,sidelen=128,neighbor=1):
    size=len(seeds[0])
    disk_size=(neighbor,sidelen,sidelen)
    cubes=[img3D[tuple(slice(_r-(_p//2),_r+_p-(_p//2)) for _r,_p in zip(r,disk_size))] for r in zip(*seeds)]
    cubes=np.array(cubes)
    return cubes


class Patch:
    def __init__(self, tomo):
        self.sp = tomo.shape
        self.tomo = tomo
    
    def to_patches(self,sidelen=128,overlap_rate = 0.25,neighbor=5):
        effect_len = int(sidelen * (1-overlap_rate))
        n1 = (self.sp[1] - sidelen)//effect_len + 3
        n2 = (self.sp[2] - sidelen)//effect_len + 3 # changed from +2 -> +3  to pad more
        n0 = self.sp[0]
        pad_len1 = (n1-1)*effect_len + sidelen - self.sp[1]
        pad_len2 = (n2-1)*effect_len + sidelen - self.sp[2]
        tomo_padded = np.pad(self.tomo,((neighbor//2, neighbor-neighbor//2),
                                        (pad_len1//2,pad_len1 - pad_len1//2),
                                        (pad_len2//2,pad_len2 - pad_len2//2)),'symmetric')
        patch_list = []
        print('padded shape',tomo_padded.shape)
        for k in range(neighbor//2,neighbor//2+self.sp[0]):
            for i in range(n1):
                for j in range(n2):
                    one_patch = tomo_padded[k-neighbor//2:k+neighbor-neighbor//2,
                                            i*effect_len:i * effect_len + sidelen,
                                            j*effect_len:j * effect_len + sidelen]
                    
                    patch_list.append(one_patch)
        print('one patch shape',one_patch.shape)
        self.n12 = (n1,n2)
        self.sidelen = sidelen
        self.effect_len = effect_len
        self.neighbor = neighbor
        self.padded_dim = tomo_padded.shape

        return patch_list

    def restore_tomo(self,patch_list,neighbor=1):
        (n1,n2) = self.n12
        sidelen = self.sidelen
        effect_len = self.effect_len
        sp = self.sp
        half1 = (sidelen - effect_len)//2
        half2 = sidelen - effect_len - half1
        
        tomo_padded = np.zeros((sp[0] + neighbor-neighbor%2,self.padded_dim[1],self.padded_dim[2]))
        for k in range(neighbor//2,neighbor//2+self.sp[0]):
            for i in range(n1):
                for j in range(n2):
                    one_patch = tomo_padded[ k-neighbor//2:k-neighbor//2+neighbor,
                                 i*effect_len:i * effect_len + sidelen,
                                j*effect_len:j * effect_len + sidelen]
                    
                    tomo_padded[ k-neighbor//2:k-neighbor//2+neighbor,
                                 i*effect_len + half1 :i * effect_len + sidelen - half2,
                                j*effect_len + half1 : j * effect_len + sidelen - half2] += \
                                    patch_list[(k-neighbor//2)*n1*n2 + i*n2 + j][:,half1:-half2,half1:-half2] 

        pad_len1 = (n1-1)*effect_len + sidelen - self.sp[1]
        pad_len2 = (n2-1)*effect_len + sidelen - self.sp[2]
        restored_tomo = tomo_padded[neighbor//2 : neighbor//2+self.sp[0],
                                pad_len1//2  : pad_len1//2 + self.sp[1],
                                pad_len2//2  : pad_len2//2 + self.sp[2],
                                ]

        return restored_tomo