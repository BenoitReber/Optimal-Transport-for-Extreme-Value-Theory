import torch
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

# Function to create scatter plots with specific styling
plotp = lambda x,col: plt.scatter(x[0,:], x[1,:], s=100, edgecolors="k", c=col, linewidths=1)

def plotr(points, 
        extr_contour, 
        hallin_contours, 
        b, 
        beta, 
        alpha, 
        threshold = 0.2, 
        axis = None,
        color = ["b", "r", "y"],
        norm_max = None,
        rescale = False,
        p = 0.005,
        ):
    """
    Plot points and contours with optional rescaling.
    
    Parameters
    ----------
    points : array or list of arrays
        The points to be plotted.
    extr_contour : array
        Extremal contour data.
    hallin_contours : list of arrays
        Hallin contours to be plotted.
    b : float
        Scaling parameter.
    beta : float
        Parameter for contour calculation.
    alpha : float
        Parameter for rescaling.
    threshold : float, optional
        Threshold for splitting contours, default is 0.2.
    axis : list, optional
        Custom axis limits [xmin, xmax, ymin, ymax].
    color : list, optional
        Colors for different elements, default is ["b", "r", "y"].
    norm_max : float, optional
        Maximum norm for rescaling.
    rescale : bool, optional
        Whether to rescale the data, default is False.
    p : float, optional
        Proportion of points to ignore when calculating ranges, default is 0.005.
    
    Returns
    -------
    None
        Displays the plot using plt.show().
    """

    plt.figure(figsize=(10,10))

    max_range = []
    min_range = []

    if norm_max is not None and rescale:
        def rescaling(t):
            norms = np.sqrt( ((t)**2).sum(axis=0) )
            dir = t / ( norms + 0.000000001)
            return (norms**(alpha/4)) * dir / ( norm_max**(alpha/4) )
        #rescaling = lambda t: ( np.abs(t) )**(alpha/4) * (t / (np.abs(t)+0.0000001)) / ( norm_max**(alpha/4) )
    elif norm_max is None and rescale:
        def rescaling(t):
            norms = np.sqrt( ((t)**2).sum(axis=0) )
            dir = t / ( norms + 0.000000001)
            return (norms**(alpha/4)) * dir
        #rescaling = lambda t: ( np.abs(t) )**(alpha/4) * (t / (np.abs(t)+0.0000001))
    else:
        rescaling = lambda t: t

    if type(points) == list:
        for i, point in enumerate(points):
            ignored = int( p * point.shape[1])

            max_range.append( max(
                                np.max( np.sort(point[0, :])[:(-max(1,ignored))] ),
                                np.max( np.sort(point[1, :])[:(-max(1,ignored))] ))
                            )

            min_range.append( min(
                                np.min( np.sort(point[0, :])[ignored:] ),
                                np.min( np.sort(point[1, :])[ignored:] ))
                            )

            plotp( rescaling(point) , color[i])
    else:
        point = points
        ignored = int( p * point.shape[1])
        max_range.append( max(
                                np.max( np.sort(point[0, :])[:(-max(1,ignored))] ),
                                np.max( np.sort(point[1, :])[:(-max(1,ignored))] ))
                            )

        min_range.append( min(
                                np.min( np.sort(point[0, :])[ignored:] ),
                                np.min( np.sort(point[1, :])[ignored:] ))
                            )
        plotp( rescaling(points), color[0])

    if len(extr_contour) > 0:

        dist_subsequent = np.abs( extr_contour[:,:-1] - extr_contour[:,1:] ).sum(axis=0)
        splits = np.argwhere(dist_subsequent > threshold)
        last = 0

        final_evt_contour = b * (1-beta)**(-1/alpha) * extr_contour

        max_range.append( max( 
                            np.max( final_evt_contour[0, :] ),
                            np.max( final_evt_contour[1, :] ))
                        ) 
            
        min_range.append( min( 
                            np.min( final_evt_contour[0, :] ),
                            np.min( final_evt_contour[1, :] ))
                        )

        for index in range(splits.shape[0] + 1):
            if index == splits.shape[0] :
                split = final_evt_contour.shape[1] + 1
            else:
                split = splits[index,0]
            
            tmp = np.zeros((2,(final_evt_contour[:,last:(split)]).shape[1] + 2))
            tmp[:,1:-1] = rescaling( final_evt_contour[:,last:(split)] )
            plt.plot(tmp[0,:], tmp[1,:], 'y')
            last = split + 1

    for hallin_contour in hallin_contours:

        max_range.append( max(
                            np.max( hallin_contour[0, :] ),
                            np.max( hallin_contour[1, :] ))
                        )

        min_range.append( min(
                            np.min( hallin_contour[0, :] ),
                            np.min( hallin_contour[1, :] ))
                        )

        rescaled_contour = rescaling(hallin_contour)
        tmp = np.zeros((rescaled_contour.shape[1]+2,2))
        tmp[1:-1,:] = rescaled_contour.T
        plt.plot(tmp[:,0], tmp[:,1], color[-1])
    
    if axis is not None:
        plt.axis(axis)
    else:
        max_range = rescaling( max(max_range) )
        min_range = rescaling( min(min_range) ) #min(min_range) * rescaling( abs( min(min_range) ) ) / min(min_range) if min(min_range) < 0 else 0
        if min_range < 0:
            max_range = max(max_range, np.abs(min_range))
            min_range = - max_range
        else:
            min_range = 0
        plt.axis([min_range, max_range, min_range, max_range])

    plt.show()

def generate_reference_data(Y, theta_min = 0, theta_max = 2 * np.pi):
    """
    Generate reference data by transforming points to polar coordinates.
    
    Parameters
    ----------
    Y : array
        Input data array.
    theta_min : float, optional
        Minimum angle in radians, default is 0.
    theta_max : float, optional
        Maximum angle in radians, default is 2π.
    
    Returns
    -------
    array
        Transformed reference data.
    """
    n = Y.shape[1]
    Theta = np.random.uniform(theta_min, theta_max ,size=n)
    tmp = np.vstack((np.cos(Theta), np.sin(Theta).T)).T
    rad = np.sqrt( np.sum(Y**2,axis=0) ) 
    return rad * tmp.T

def distmat(x,y):
    """
    Compute the distance matrix between two sets of points.
    
    Parameters
    ----------
    x : array
        First set of points.
    y : array
        Second set of points.
    
    Returns
    -------
    array
        Distance matrix between x and y.
    """
    return np.sum(x**2,0)[:,None] + np.sum(y**2,0)[None,:] - 2*x.transpose().dot(y)

def compute_d(costs,n):
    """
    Compute dynamic programming matrix for optimal transport.
    
    Parameters
    ----------
    costs : array
        Cost matrix.
    n : int
        Size of the problem.
    
    Returns
    -------
    array
        Dynamic programming matrix.
    """
    d = np.ones((n+1,n)) * np.inf
    d[0,0] = 0

    for k in range(0,n):
        d[k+1,:] = (d[k,:] + costs.T).min(axis=1)

    return d

def compute_costs(X, Y):
    """
    Compute cost matrix between two sets of points.
    
    Parameters
    ----------
    X : array
        First set of points.
    Y : array
        Second set of points.
    
    Returns
    -------
    array
        Cost matrix with diagonal set to infinity.
    """
    n = X.shape[1]
    tmp = X.T @ Y # (x_i | y_j)
    costs = (tmp.diagonal() - tmp.T).T
    del(tmp)
    costs[ np.eye(n)==1 ] = np.inf
    return costs

def compute_params(X, Y):
    """
    Compute optimal transport parameters between two sets of points
    following Hallin et al.(2021).
    
    Parameters
    ----------
    X : array
        First set of points.
    Y : array
        Second set of points.
    
    Returns
    -------
    tuple
        (epsilon_star, psi)
    """
    n = X.shape[1]
    
    costs = compute_costs(X, Y)

    d = compute_d(costs,n)
    
    epsilon_star = ( ( ((d[n,:] - d)[:-1,:]).T / (n-np.arange(0,n)) ).T ).max(axis=0).min()

    d_bis = compute_d(costs - epsilon_star, n)
    shortest_paths = d_bis[:-1,:].min(axis=0)
    psi = - shortest_paths

    return epsilon_star, psi

def check_optimality_cdtn(costs, psi, epsilon_star, eps):
    """
    Check optimality conditions for optimal transport problem.
    
    Parameters
    ----------
    costs : array
        Cost matrix.
    psi : array
        Dual variable.
    epsilon_star : float
        Optimal smoothing parameter.
    eps : float
        Tolerance parameter.
    
    Returns
    -------
    tuple
        (spread, tmp) where spread contains the optimality gaps and
        tmp is a binary matrix indicating satisfied conditions.
    """
    n = costs.shape[0]
    tmp = np.zeros((n,n))
    spread = np.zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            spread[i,j] = ( costs[i,j] - (psi[i] - psi[j] + epsilon_star) )
            tmp[i,j] = 1 if spread[i,j] >= - eps else 0
    return spread, tmp

def check_optimality_cdtn_bis(X, Y, psi):
    """
    Alternative check for optimality conditions using point sets directly.
    
    Parameters
    ----------
    X : array
        First set of points.
    Y : array
        Second set of points.
    psi : array
        Dual variable.
    
    Returns
    -------
    tuple
        (spread, tmp_bis) where spread contains the optimality gaps and
        tmp_bis is a binary array indicating satisfied conditions.
    """
    n = X.shape[1]
    tmp_bis = np.zeros(n)
    spread = np.zeros(n)
    for i in range(0,n):
        x = X[:,i]
        y = Y[:,i]
        spread[i] = ( (np.dot( x, Y) - psi).max() - (np.dot( x, y) - psi[i]) )
        tmp_bis[i] = 1 if spread[i] == 0 else 0
    return spread, tmp_bis

def smooth_T(x, Y, epsilon, psi, norm_max, lr=None, steps=None):
    """
    Compute a smoothed optimal transport map that is cyclically 
    monotone following Hallin et al.(2021) using gradient descent.
    
    Parameters
    ----------
    x : array
        New point.
    Y : array
        Target points.
    epsilon : float
        Smoothing parameter.
    psi : array
        Dual variable.
    norm_max : float
        Normalization factor.
    lr : float, optional
        Learning rate for optimizer, defaults to epsilon if None.
    steps : int, optional
        Number of optimization steps, defaults to Y.shape[1] if None.
    
    Returns
    -------
    array
        The smoothed optimal transport vector.
    """
    psi = torch.tensor(psi,requires_grad=False)
    Y = torch.tensor(Y,requires_grad=False)

    x = torch.tensor(x / norm_max, dtype=torch.float64, requires_grad=False)

    y_0 = torch.tensor( x.clone().detach(), dtype=torch.float64, requires_grad=True)
    if lr is not None:
        optimizer = torch.optim.SGD([y_0], lr=lr)
    else:
        optimizer = torch.optim.SGD([y_0], lr=epsilon)
    steps = Y.shape[1] if steps is None else steps
    for i in range(steps):
        optimizer.zero_grad()
        phi_y = (torch.matmul( y_0, Y) - psi).max()
        loss = (phi_y + (y_0-x).pow(2).sum() / (2*epsilon))
        loss.backward()
        optimizer.step()

    return (norm_max * (x-y_0) / epsilon).detach().numpy()