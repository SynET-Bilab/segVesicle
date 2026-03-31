import numpy as np
import math 
import scipy
from scipy.sparse import csr_matrix
from numpy.linalg import eig, inv
from scipy.optimize import minimize


def _ellipsoid_fit_n(X, alpha=1.5, gamma=0.1, lambda_reg=1000.0):
    
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
        return L @ L.T
    
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


# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
def _ellipsoid_fit_llsq(X):
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                 x * x + z * z - 2 * y * y,
                 2 * x * y,
                 2 * x * z,
                 2 * y * z,
                 2 * x,
                 2 * y,
                 2 * z,
                 1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)
    
    return center, evecs, radii


def ellipsoid_fit(  X, 
                    alpha=0.3,
                    n_iter=10,
                    alpha_n=1.5,
                    gamma=0.1,
                    lambda_reg=1000.0,
                    use_LLSQ_regularization=False,
                    use_sampson_regularization=False
                ):
    """
    """
    # determine method
    if use_sampson_regularization:
        if use_LLSQ_regularization:
            print('Conflicting regularization methods specified. Use Sampson regularization.')
        return _ellipsoid_fit_n(X, alpha_n, gamma, lambda_reg)
    
    center0, evecs0, radii0 = _ellipsoid_fit_llsq(X)

    if not use_LLSQ_regularization:
        return center0, evecs0, radii0
    n = len(X)

    def _compute_rss(X, center, evecs, radii):
        """
        """
        diff = X - center
        cloud_r = diff @ evecs.T
        radii_safe = np.maximum(np.abs(radii), 1e-6)
        cloud_n = cloud_r / radii_safe
        d = np.sqrt(np.sum(cloud_n**2, axis=1))
        return np.mean((d - 1.0)**2)
    
    centroid = np.median(X, axis=0)
    dists = np.linalg.norm(X - centroid, axis=1)
    trim_idx = np.argsort(dists)[:int(0.8 * n)]
    try:
        center, evecs, radii = _ellipsoid_fit_llsq(X[trim_idx])
    except Exception:
        center, evecs, radii = center0, evecs0, radii0

    best_center, best_evecs, best_radii = center, evecs, radii
    best_rss = _compute_rss(X, center, evecs, radii)

    for iteration in range(n_iter):
        diff = X - center
        cloud_r = diff @ evecs.T
        radii_safe = np.maximum(np.abs(radii), 1e-6)
        cloud_n = cloud_r / radii_safe
        d = np.sqrt(np.sum(cloud_n**2, axis=1))
        residuals = np.abs(d - 1.0)

        mad = np.median(residuals)
        
        progress = iteration / max(n_iter - 1, 1)
        threshold = max(alpha, mad * (5.0 - 3.0 * progress))

        inlier_mask = residuals < threshold
        n_inliers = np.sum(inlier_mask)

        min_points = max(10, int(0.6 * n))
        if n_inliers < min_points:
            sorted_idx = np.argsort(residuals)
            inlier_mask = np.zeros(n, dtype=bool)
            inlier_mask[sorted_idx[:min_points]] = True

        try:
            center_new, evecs_new, radii_new = _ellipsoid_fit_llsq(X[inlier_mask])
            if np.all(radii_new > 0) and np.all(np.isfinite(radii_new)):
                center, evecs, radii = center_new, evecs_new, radii_new
                rss = _compute_rss(X[inlier_mask], center, evecs, radii)
                if rss < best_rss or iteration == 0:
                    best_center, best_evecs, best_radii = center, evecs, radii
                    best_rss = rss
        except Exception:
            pass

    return best_center, best_evecs, best_radii


def ellipse_fit(x, y, Zc):
    """
    ellipse fitting by least square method
    http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html
    """
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
