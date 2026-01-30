import numpy as np
import math 
import scipy
from scipy.sparse import csr_matrix
from numpy.linalg import eig, inv
from scipy.optimize import minimize

def ellipsoid_fit(X, alpha=1.5, gamma=0.1, lambda_reg=1000.0):
    # 标归一化
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-6] = 1.0 
    points_norm = (X - mean) / std
    
    # 2. 初始化参数
    center_init = np.mean(points_norm, axis=0)
    A_init = np.diag([1.0, 1.0, 1.0])  
    
    # 从对称矩阵A提取下三角参数 (6个独立参数)
    def matrix_to_params(A):
        return np.array([A[0,0], A[1,0], A[1,1], A[2,0], A[2,1], A[2,2]])
    
    # 从参数构建对称正定矩阵 (通过下三角分解)
    def params_to_matrix(params):
        L = np.array([
            [params[0], 0, 0],
            [params[1], params[2], 0],
            [params[3], params[4], params[5]]
        ])
        return L @ L.T  # 保证对称正定
    
    # 3. 定义目标函数 (带正则项)
    def objective(params, pts):
        center = params[:3]
        A = params_to_matrix(params[3:])
        
        # 计算代数距离 F(x) = (x-c)^T A (x-c) - 1
        diff = pts - center
        F_vals = np.einsum('ni,ij,nj->n', diff, A, diff) - 1.0
        
        # 计算梯度范数 ||∇F|| = 2 * ||A(x-c)||
        grad_norm = 2 * np.linalg.norm((A @ diff.T).T, axis=1)
        
        # 加权代数距离 (近似几何距离): |F| / ||∇F||
        epsilon = 1e-8
        weights = 1.0 / np.maximum(grad_norm, epsilon)
        weighted_dist = np.abs(F_vals) * weights
        
        # 计算原始损失 (最小二乘)
        loss = np.sum(weighted_dist**2)
        
        # 计算正则项
        d_i = weighted_dist  # 点到椭球表面的距离
        
        # sigmoid函数 σ(d_i) = 1/(1+e^(-((d_i-α)/γ)))
        # 数值稳定实现：限制输入范围防止exp溢出
        sigmoid_input = (d_i - alpha) / gamma
        sigmoid_input = np.clip(sigmoid_input, -100, 100)  # 防止数值溢出
        sigma = 1.0 / (1.0 + np.exp(-sigmoid_input))
        
        # 正则项: λ·∑〖σ^2 (d_i)〗
        regularization = lambda_reg * np.sum(sigma**2)
        
        return loss + regularization
    
    # 4. 设置初始参数和边界
    params_init = np.concatenate([
        center_init, 
        matrix_to_params(A_init)
    ])
    
    # 边界设置
    bounds = [
        (np.min(points_norm[:,0]), np.max(points_norm[:,0])),
        (np.min(points_norm[:,1]), np.max(points_norm[:,1])),
        (np.min(points_norm[:,2]), np.max(points_norm[:,2])),
        (1e-6, None),  
        (None, None),   
        (1e-6, None),  
        (None, None),   
        (None, None),   
        (1e-6, None)   
    ]
    
    # 5. L-BFGS-B
    result = minimize(
        fun=objective,
        x0=params_init,
        args=(points_norm,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
    )
    
    # 6. 从优化结果提取参数
    opt_params = result.x
    center_norm = opt_params[:3]
    A_norm = params_to_matrix(opt_params[3:])
    
    # 7. 转换回原始坐标系
    center_orig = center_norm * std + mean
    
    # 计算原始坐标系的变换矩阵
    D = np.diag(std)  # 缩放矩阵
    A_orig = np.linalg.inv(D) @ A_norm @ np.linalg.inv(D)
    
    # 8. 从矩阵A提取几何参数 (特征值分解)
    eigvals, eigvecs = np.linalg.eigh(A_orig)
    
    # 椭球标准方程: (x-c)^T A (x-c) = 1
    # 半轴长度 a=1/sqrt(λ1), b=1/sqrt(λ2), c=1/sqrt(λ3)
    radii = 1.0 / np.sqrt(np.maximum(eigvals, 1e-10))
    
    # 9. 确保特征向量形成右手坐标系
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] = -eigvecs[:, 2]
    
    # 10. 按半轴长度降序排列 (可选，保持与原方法一致)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    eigvecs = eigvecs[:, order]
    
    return center_orig, eigvecs, radii

def ellipse_fit(x, y, Zc):

    x=np.array(x,dtype=np.double)
    y=np.array(y,dtype=np.double)
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  eig(np.dot(inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]

    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    center = np.array([Zc,y0,x0])

    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    radii = np.array([0.01, res2, res1])

    angle = 0.5*np.arctan(2*b/(a-c))
    evecs = np.array([[1,0,0], [0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
    #zyx.dot(mat)
    return center, evecs, radii


def ellipse_fit_n(x, y, Zc):
    def sigmoid(d, a_threshold, gamma):
        """向量化sigmoid函数"""
        return 1 / (1 + np.exp(-(d - a_threshold) / gamma))

    def compute_distances_vectorized(u_all, v_all, a, b, max_iter=20):
        """向量化近似距离计算 (基于椭圆代数距离近似几何距离)"""
        # 代数投影近似公式 (避免牛顿迭代)
        phi = np.arctan2(v_all * a, u_all * b)  
        u_proj = a * np.cos(phi)
        v_proj = b * np.sin(phi)
        distances = np.sqrt((u_all - u_proj)**2 + (v_all - v_proj)**2)
        return distances

    def objective_vectorized(params, data, a_threshold, gamma):
        """向量化目标函数"""
        h, k, theta, a, b = params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # 批量坐标变换 (矩阵运算)
        dx = data[:, 0] - h
        dy = data[:, 1] - k
        u_all = dx * cos_theta + dy * sin_theta 
        v_all = -dx * sin_theta + dy * cos_theta
        
        # 向量化距离计算
        distances = compute_distances_vectorized(u_all, v_all, a, b)
        
        # 向量化权重计算
        weights = sigmoid(distances, a_threshold, gamma)
        total = np.sum(distances**2) + 1000*np.sum(weights**2)
        return total
    
    # 转换输入为numpy数组
    x = np.array(x, dtype=np.double)
    y = np.array(y, dtype=np.double)
    data = np.column_stack((x, y))
    
    # 设置初始参数 [h, k, theta, a, b]
    # h, k: 椭圆中心坐标
    # theta: 旋转角度
    # a, b: 半长轴和半短轴
    initial_params = [
        np.mean(x), 
        np.mean(y), 
        0, 
        0.5 * (np.max(x) - np.min(x)), 
        0.5 * (np.max(y) - np.min(y))
    ]
    
    # 执行优化
    result = minimize(
        objective_vectorized, 
        initial_params, 
        args=(data, 1.5, 0.1),
        method='L-BFGS-B', 
        bounds=[
            (None, None),    
            (None, None),   
            (None, None),    
            (1e-6, None),    
            (1e-6, None)     
        ],
        options={'maxiter': 200, 'ftol': 1e-8}
    )
    
    # 提取优化后的参数
    h_opt, k_opt, angle, a_opt, b_opt = result.x
    
    # 构建3D椭球参数
    center = np.array([Zc, k_opt, h_opt])
    evecs = np.array([
        [1, 0, 0], 
        [0, np.sin(angle), np.cos(angle)],
        [0, np.cos(angle), -np.sin(angle)]
    ])
    
    # 半轴长度 [z半轴, y半轴, x半轴]
    # z方向半轴设为0.01，表示扁平椭球
    radii = np.array([0.01, a_opt, b_opt])
    
    return center, evecs, radii
