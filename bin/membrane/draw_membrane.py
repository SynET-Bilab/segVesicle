""" 

References: https://github.com/SynET-Bilab/SegSynMembrane
"""
import numpy as np
import tslearn.metrics

from .bspline import Curve

def points_deduplicate(pts):
    """ Deduplicate int points while retaining the order according to the first appearances.

    Args:
        pts (np.ndarray): Array of points with shape=(npts,dim).

    Returns:
        pts_dedup (np.ndarray): Deduplicated array of points, with shape=(npts_dedup,dim).
    """
    # round
    pts = np.round(pts).astype(int)
    # convert to tuple, deduplicate using dict
    pts = [tuple(pt) for pt in pts]
    pts_dedup = dict.fromkeys(pts).keys()
    # convert to np.ndarray
    pts_dedup = np.array(list(pts_dedup))
    return pts_dedup

def wireframe_length(pts_net, axis=0):
    """ Calculate total lengths of wireframe along one axis.

    Input can be either net-shaped points or flattened points.

    Args:
        pts_net (np.ndarray): Net-shaped nu*nv points arranged in net-shape, with shape=(nu,nv,dim). Or flattened points with shape=(nu,dim).
        axis (int): 0 for u-direction, 1 for v-direction.

    Returns:
        wires (np.ndarray): [len0,len1,...]. nv elements if axis=u, nu elemens if axis=v.
    """
    # A, B - axes
    # [dz,dy,dx] along A for each B
    diff_zyx = np.diff(pts_net, axis=axis)
    # len of wire segments along A for each B
    segments = np.linalg.norm(diff_zyx, axis=-1)
    # len of wire along A for each B
    wires = np.sum(segments, axis=axis)
    return wires

def match_two_contours(pts1, pts2, closed=False):
    """ Match the ordering of points in contour2 to that of contour1.

    Points in the contours are assumed sorted.
    Dynamic time warping (dtw) is used to match the points and measure the loss.
    For open contours, contour1 is compared with the forward and reverse ordering of contour2.
    For closed contours, contour1 is compared with the forward and reverse ordering of contour2, as well as its rolls.
    The ordering of contour2 with the lowest loss is adopted.
    
    Args:
        pts1, pts2 (np.ndarray): Points in contour1,2, each with shape=(npts,dim).
        closed (bool): Whether the contour is closed or not.

    Returns:
        pts2_best (np.ndarray): Same points as in pts2, but in the best order that matches pts1.
        path_best (list of 2-tuples): Pairing between contours, [(i10,i20),(i11,i21),...], where pts1[i1j] and pts2_best[i2j] are matched pairs.
    """
    # generate candidate reordered pts2
    pts2_rev = pts2[::-1]
    # if open contour: original + reversed
    if not closed:
        pts2_arr = [pts2, pts2_rev]
    # if close contour: original + rolls + reversed + rolls of reversed
    else:
        n2 = len(pts2)
        pts2_arr = (
            [np.roll(pts2, i, axis=0) for i in range(n2)]
            + [np.roll(pts2_rev, i, axis=0) for i in range(n2)]
        )

    # calc dtw for each candidate reordered pts2
    path_arr = []
    loss_arr = []
    for pts2_i in pts2_arr:
        path_i, loss_i = tslearn.metrics.dtw_path(pts1, pts2_i)
        path_arr.append(path_i)
        loss_arr.append(loss_i)

    # select the best order
    i_best = np.argmin(loss_arr)
    pts2_best = pts2_arr[i_best]
    path_best = path_arr[i_best]
    return pts2_best, path_best

def interpolate_contours_alongz(zyx, closed=False):
    """ Interpolate contours along z direction.
    
    The input contour is sparsely sampled in z direction.
    The contour in each xy-plane is assumed ordered.
    Contours at different z's are first order-matched using dynamic time warping.
    Interpolation is then performed at the missing z's.
    
    Args:
        zyx (np.ndarray): Points in the contour, with shape=(npts,3).
        closed (bool): Whether the contour is closed or not.

    Returns:
        zyx_interp (np.ndarray): Points in the interpolated contour, with shape=(npts_interp,3).
    """
    zyx_ct = np.round(zyx).astype(int)
    z_uniq = sorted(np.unique(zyx_ct[:, 0]))
    z_ct = zyx_ct[:, 0]
    zyx_dict = {}

    for z1, z2 in zip(z_uniq[:-1], z_uniq[1:]):
        # get correspondence between two given contours
        if z1 not in zyx_dict:
            zyx_dict[z1] = zyx_ct[z_ct == z1]
        zyx1 = zyx_dict[z1]
        zyx2_raw = zyx_ct[z_ct == z2]
        zyx2, path12 = match_two_contours(zyx1, zyx2_raw, closed=closed)
        zyx_dict[z2] = zyx2

        # linear interpolation on intermediate z's
        for zi in range(z1+1, z2):
            zyx_dict[zi] = np.array([
                ((z2-zi)*zyx1[p[0]] + (zi-z1)*zyx2[p[1]])/(z2-z1)
                for p in path12
            ])

    # round to int
    zyx_interp = np.concatenate([
        zyx_dict[zi]
        for zi in range(z_uniq[0], z_uniq[-1]+1)
    ], axis=0)

    return zyx_interp

def interpolate_contours_alongxy(zyx, degree=2):
    """ Interpolate open contours in xy-planes.
    
    Args:
        zyx (np.ndarray): Points in the contour, with shape=(npts,3).
        degree (int): The degree for bspline interpolation.

    Returns:
        zyx_interp (np.ndarray): Points in the interpolated contour, with shape=(npts_interp,3).
    """
    # deduplicate
    zyx = points_deduplicate(zyx)
    
    # iterate over z's
    z_arr = zyx[:, 0]
    zyx_interp = []
    for z_i in np.unique(z_arr):
        # fit yx points
        yx_i = zyx[z_arr==z_i][:, 1:]
        fit = Curve(degree).interpolate(yx_i)

        # evaluate at dense parameters, then deduplicate
        n_eval = int(2*wireframe_length(yx_i))
        yx_interp_i = fit(np.linspace(0, 1, n_eval))
        yx_interp_i = points_deduplicate(yx_interp_i)

        # stack yx with z
        z_ones_i = z_i*np.ones((len(yx_interp_i), 1))
        zyx_interp_i = np.concatenate([
            z_ones_i, yx_interp_i
        ], axis=1)
        
        zyx_interp.append(zyx_interp_i)
    
    zyx_interp = np.concatenate(zyx_interp, axis=0)
    return zyx_interp

def draw_membrane(points_guide, interp_degree=2):
    """
    处理模型指导点，进行插值和去重。

    Args:
        model_guide (np.ndarray): 指导线的模型点，形状为 (npts, 3)，每个点为 [z, y, x]。
        interp_degree (int): 在 xy 方向进行 bspline 插值的阶数。

    Returns:
        guide (np.ndarray): 经过插值和去重后的指导线点，形状为 (npts_interp, 3)。
    """
    # 去重
    guide_raw = points_deduplicate(points_guide)

    # 移除每个 z 切片中点数 <=1 的情况
    guide = []
    for z in np.unique(guide_raw[:, 0]):
        guide_raw_z = guide_raw[guide_raw[:, 0] == z]
        if len(guide_raw_z) > 1:
            guide.append(guide_raw_z)
    if len(guide) == 0:
        raise ValueError("在所有 z 切片中，指导线点数均 <=1。请检查输入的 points_guide 数据。")
    guide = np.concatenate(guide, axis=0)

    # 沿 z 方向进行插值
    guide = interpolate_contours_alongz(guide, closed=False)

    # # 沿 xy 平面进行插值
    guide = interpolate_contours_alongxy(guide, degree=interp_degree)

    # 再次去重，确保指导线点唯一
    guide = points_deduplicate(guide)

    return guide

def process_bottom_guide(bottom_guide, interp_degree=2, n_line=30):
    """
    根据输入的一批点（bottom_guide），生成一个按钮曲面点集：
      1. bottom_guide 只含有两个 z 值，其中一个 z 值对应多点（平面），另一个 z 值对应 1 点（按钮点）。
      2. 将同一平面的一批点进行曲线（闭合）插值，得到一条光滑闭合曲线。
      3. 对闭合曲线去重，然后对曲线上的每个点与按钮点进行线性插值，得到新的曲面点集。
      4. 对曲面点集去重并输出。

    Args:
        bottom_guide (np.ndarray): 输入点集，形状 (npts, 3)，只包含 2 个唯一的 z 值。
                                   其中一个 z 值对应多点（平面），另一个 z 值对应 1 点（按钮）。
        interp_degree (int): 在 xy 平面做 B-spline 插值的阶数，默认 2。
        n_line (int): 每个曲线点到按钮点之间的插值步数，默认 30。

    Returns:
        surface_points (np.ndarray): 生成的曲面点集，形状 (m, 3)。
    """
    bottom_guide = np.asarray(bottom_guide)
    if bottom_guide.shape[1] != 3:
        raise ValueError("bottom_guide 必须是形如 (npts,3) 的数组，每行代表 [z,y,x].")

    # 1. 解析两种 z 值
    z_unique = np.unique(bottom_guide[:, 0])
    if len(z_unique) != 2:
        raise ValueError("bottom_guide 必须只包含 2 个唯一的 z 值，一个平面多点，一个按钮单点。")

    # 根据点数判断哪个是按钮点、哪个是平面点
    z1, z2 = z_unique
    plane_z_candidates = bottom_guide[bottom_guide[:, 0] == z1]
    bottom_z_candidates = bottom_guide[bottom_guide[:, 0] == z2]
    # 如果哪个 z 值只有 1 个点，则它就是按钮 z
    # 哪个 z 值超过 1 个点则是平面 z
    if len(plane_z_candidates) == 1 and len(bottom_z_candidates) > 1:
        # 需要对调
        plane_z_candidates, bottom_z_candidates = bottom_z_candidates, plane_z_candidates
        z1, z2 = z2, z1
    elif len(plane_z_candidates) == 1 and len(bottom_z_candidates) == 1:
        raise ValueError("两个 z 值各自只包含 1 个点，无法生成平面批量点和按钮点。")
    elif len(plane_z_candidates) == 1 and len(bottom_z_candidates) > 1:
        # 正常情况：plane_z_candidates=1 点, 需要对调
        plane_z_candidates, bottom_z_candidates = bottom_z_candidates, plane_z_candidates
        z1, z2 = z2, z1
    # 检查一下，按钮数组必须只有1点
    if len(bottom_z_candidates) != 1:
        raise ValueError("无法唯一识别按钮点，请确保另一个 z 值只对应 1 个点。")

    plane_points = plane_z_candidates  # 多点（同一个z）
    bottom_point = bottom_z_candidates[0]  # 单点 (z,y,x)

    # 2. 对平面批量点进行曲线插值（闭合）
    #    首先去重
    plane_points = points_deduplicate(plane_points)
    if len(plane_points) < 3:
        raise ValueError("平面批量点少于 3 个，无法形成有效的闭合曲线。")

    #    让曲线首尾相连：将首个点附加到末尾，以便插值成闭合
    #    （注意：plane_points 的所有点 z 都相同，所以这样做不会影响 z 轴处理）
    plane_points_closed = np.vstack([plane_points, plane_points[0:1]])

    #    现在使用 interpolate_contours_alongxy 来做 xy 插值
    #    由于 plane_points_closed 的 z 都相同，函数会在这一层 z 上进行插值
    curve = interpolate_contours_alongxy(plane_points_closed, degree=interp_degree)
    #    再次去重
    curve = points_deduplicate(curve)

    # 3. 使用矩阵运算计算平面点的最大两两距离
    plane_yx = plane_points[:, 1:3]  # 只需要 (y, x) 坐标即可
    #   生成坐标差分矩阵 delta，形状 (N, N, 2)
    delta = plane_yx[:, None, :] - plane_yx[None, :, :]  
    #   计算距离矩阵 dist，形状 (N, N)
    dist2 = np.sum(delta**2, axis=-1)
    dist = np.sqrt(dist2)
    max_point_dis = dist.max()  # 最大距离

    #   设置插值步数 n_line，最少为 2，避免插值过少
    n_line = max(2, int(max_point_dis // 2))

    # 4. XY 线性插值, Z 采用先快后慢(√t)
    plane_z = curve[0, 0]     # 平面 z
    bottom_z = bottom_point[0]
    surface_points_list = []

    for pt in curve:
        # pt = [z0, y0, x0]
        # bottom_point = [zb, yb, xb]
        z0, y0, x0 = pt
        dz = bottom_z - z0
        dy = bottom_point[1] - y0
        dx = bottom_point[2] - x0

        for t in np.linspace(0, 1, n_line):
            # Z 非线性插值: z(t) = z0 + sqrt(t)*(dz)
            zt = z0 + dz * (t ** 0.25)
            # XY 线性插值: y(t)=y0 + t*dy, x(t)=x0 + t*dx
            yt = y0 + t*dy
            xt = x0 + t*dx

            interp_pt = [zt, yt, xt]
            surface_points_list.append(interp_pt)

    surface_points = np.array(surface_points_list)
    #   去重
    surface_points = points_deduplicate(surface_points)

    return surface_points