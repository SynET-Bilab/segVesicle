import numpy as np
import math 
import scipy
from scipy.sparse import csr_matrix
from numpy.linalg import eig, inv
from scipy.optimize import minimize

def ellipsoid_fit(X, alpha=1.5, gamma=0.1, lambda_reg=1000.0):
    
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std < 1e-6] = 1.0 
    points_norm = (X - mean) / std
    
    center_init = np.mean(points_norm, axis=0)
    A_init = np.diag([1.0, 1.0, 1.0])  
    
    def matrix_to_params(A):
        return np.array([A[0,0], A[1,0], A[1,1], A[2,0], A[2,1], A[2,2]])
    
    def params_to_matrix(params):
        L = np.array([
            [params[0], 0, 0],
            [params[1], params[2], 0],
            [params[3], params[4], params[5]]
        ])
        return L @ L.T  # 保证对称正定
    
    def objective(params, pts):
        center = params[:3]
        A = params_to_matrix(params[3:])
        
        # F(x) = (x-c)^T A (x-c) - 1
        diff = pts - center
        F_vals = np.einsum('ni,ij,nj->n', diff, A, diff) - 1.0
        
        # ||∇F|| = 2 * ||A(x-c)||
        grad_norm = 2 * np.linalg.norm((A @ diff.T).T, axis=1)
        
        # |F| / ||∇F||
        epsilon = 1e-8
        weights = 1.0 / np.maximum(grad_norm, epsilon)
        weighted_dist = np.abs(F_vals) * weights
        
        loss = np.sum(weighted_dist**2)
        d_i = weighted_dist
        
        sigmoid_input = (d_i - alpha) / gamma
        sigmoid_input = np.clip(sigmoid_input, -100, 100)
        sigma = 1.0 / (1.0 + np.exp(-sigmoid_input))
        
        # λ·∑〖σ^2 (d_i)〗
        regularization = lambda_reg * np.sum(sigma**2)
        
        return loss + regularization
    
    params_init = np.concatenate([
        center_init, 
        matrix_to_params(A_init)
    ])
    
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
    
    # L-BFGS-B
    result = minimize(
        fun=objective,
        x0=params_init,
        args=(points_norm,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'ftol': 1e-8, 'gtol': 1e-8, 'disp': False}
    )
    
    opt_params = result.x
    center_norm = opt_params[:3]
    A_norm = params_to_matrix(opt_params[3:])
    
    center_orig = center_norm * std + mean
    
    D = np.diag(std)
    A_orig = np.linalg.inv(D) @ A_norm @ np.linalg.inv(D)
    
    eigvals, eigvecs = np.linalg.eigh(A_orig)
    
    # (x-c)^T A (x-c) = 1
    # a=1/sqrt(λ1), b=1/sqrt(λ2), c=1/sqrt(λ3)
    radii = 1.0 / np.sqrt(np.maximum(eigvals, 1e-10))
    
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] = -eigvecs[:, 2]
    
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
        return 1 / (1 + np.exp(-(d - a_threshold) / gamma))

    def compute_distances_vectorized(u_all, v_all, a, b, max_iter=20):
        phi = np.arctan2(v_all * a, u_all * b)  
        u_proj = a * np.cos(phi)
        v_proj = b * np.sin(phi)
        distances = np.sqrt((u_all - u_proj)**2 + (v_all - v_proj)**2)
        return distances

    def objective_vectorized(params, data, a_threshold, gamma):
        h, k, theta, a, b = params
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        dx = data[:, 0] - h
        dy = data[:, 1] - k
        u_all = dx * cos_theta + dy * sin_theta 
        v_all = -dx * sin_theta + dy * cos_theta
        
        distances = compute_distances_vectorized(u_all, v_all, a, b)
        
        weights = sigmoid(distances, a_threshold, gamma)
        total = np.sum(distances**2) + 1000*np.sum(weights**2)
        return total
    
    x = np.array(x, dtype=np.double)
    y = np.array(y, dtype=np.double)
    data = np.column_stack((x, y))
    
    initial_params = [
        np.mean(x), 
        np.mean(y), 
        0, 
        0.5 * (np.max(x) - np.min(x)), 
        0.5 * (np.max(y) - np.min(y))
    ]
    
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
    
    h_opt, k_opt, angle, a_opt, b_opt = result.x
    
    center = np.array([Zc, k_opt, h_opt])
    evecs = np.array([
        [1, 0, 0], 
        [0, np.sin(angle), np.cos(angle)],
        [0, np.cos(angle), -np.sin(angle)]
    ])
    
    radii = np.array([0.01, a_opt, b_opt])
    
    return center, evecs, radii
