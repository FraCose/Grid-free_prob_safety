import numpy as np
import cupy as cp
import matplotlib
matplotlib.use('tkagg')
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from itertools import product, combinations
import copy

# THIS REPOSITORY CONTAINS THE ALGORITHMS EXPLAINED IN THE WORK
# Cosentino, Abate, Oberhauser
# "Grid-Free Computation of Probabilistic Safety with Malliavin Calculus"

####################################################################################
# 
#  - simul_gpu and simul_gpu_anti are the functions relative to the Monte Carlo part 
# realtive to the computation of the Probability of the exit time and its derivative.
# The safe region (sphere or cube) is defined internally in these functions, however, 
# they require some date regarding the shape.
# simul_gpu_anti applies the antithetic approach to reduce the variance.
#
#  - mu_gpu, Dmu_gpu, sigma_gpu, Dsigma_gpu, inv_sigma_gpu are the functions that specifiy the model/SDE
#
#  - minimize run the GD part of the algorithms. 
#
#  - parabola approximate the local second order information of the border as explained in the work.
#
#  - In main is present the procedure to walk on the border, it should plot a figure that automatically is updated
# when new points of the border are discovered.
#
#  - The running time of the notebook could be quite long (about 1 hour). 
# 
#  - See the reference for more details. 
# 
####################################################################################

def simul_gpu(x_0, N, n, d, 
            sphere_c, sphere_r, 
            mu, sigma, Dmu, Dsigma, inv_sigma, 
            FIG=True):
    
    assert d==len(sphere_c), "The dim. of X and the dim. of the sphere do not coincide"
    
    # cp.random.seed(int(seed))
    # cp.random.randn(3)

    if FIG: 
        plt.ion()
        plt.title('Greek for safety')
        # fig = plt.figure()
        ax = plt.axes(projection='3d')
        # plt.xlabel('X_1')
        # plt.ylabel('X_2')
        # plt.yzlabel('X_3')

        # DRAW SPHERE
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)*cp.asnumpy(sphere_r)
        y = np.sin(u)*np.sin(v)*cp.asnumpy(sphere_r)
        z = np.cos(v)*cp.asnumpy(sphere_r)
        ax.plot_wireframe(x, y, z, color="r")

        # DRAW SQUARE
        # square = [[x1, x2, x3],[X1, X2, X3]]
        # square_cpu = cp.asnumpy(square)
        # for s, e in combinations(cp.array(list(product(square[:,0], square[:,1], square[:,2]))), 2):
        #     if cp.sum(s==e) > 1 : ax.plot3D(*zip(s, e), color="b")

        # plt.show(block=True) 
        plt.pause(1e-7) 
        
    X = cp.repeat(x_0.reshape(1,3), N, axis=0)
    J = cp.repeat(cp.eye(d).reshape(1,d,d), N, axis=0)
    int_tau1 = cp.zeros(N)
    H = cp.zeros((N,d))

    tau = cp.ones(N, dtype='bool')      # true if always inside the square
    tau1 = cp.ones(N, dtype='bool')     # true if t<tau1
    
    for i in range(n):
        
        ############ check if X is inside the square
        # idx_tmp = cp.logical_and(X[tau]-square[0], X[tau]<square[1])
        # idx_tmp = cp.sum(idx_tmp, 1) < 3

        ############ check if X is inside the sphere  
        dist = cp.square(X[tau]-sphere_c).sum(axis=1)
        idx_tmp =  dist > sphere_r**2
        tmp = cp.arange(N)[tau]
        tau[tmp[idx_tmp]] = False
        tau1[tmp[idx_tmp]] = False

        ############ check tau1
        tmp = cp.arange(N)[tau1]
        tmp = tmp[int_tau1[tau1]>=1.]
        tau1[tmp] = False

        ############ compute distance square
        # dist = cp.concatenate((cp.square(X[tau1]-square[0]), cp.square(X[tau1]-square[1])), axis=1)
        # dist = cp.min(dist, axis=1)         # dist is already squared

        ############ compute distance sphere
        dist = cp.square(X[tau1]-sphere_c).sum(axis=1)
        dist = cp.square(cp.sqrt(dist) - sphere_r)    # dist is already squared

        ############ compute tau1 integral
        N_star = cp.int(cp.sum(tau1))
        int_tau1[tau1] += 1 / (dist * n)
        alpha = cp.ones(N_star)
        if cp.sum(tau1)!=0:
            if cp.max(int_tau1[tau1])>1:
                tmp = cp.arange(N_star)[int_tau1[tau1]>1]
                tmp1 = cp.arange(N)[tau1]
                tmp1 = tmp1[tmp]
                alpha[tmp] = dist[tmp] * n * (1-(int_tau1[tmp1]-1/(dist[tmp]*n)))
                int_tau1[tmp1] = int_tau1[tmp1] - (1- alpha[tmp])/ (dist[tmp] * n)

        # if cp.max(int_tau1)>1.+1e-8: print("check int_tau1")
        
        N_star = cp.int(cp.sum(tau))
        DW = cp.zeros(N,d)
        DW[tau] = cp.random.randn(N_star,d)

        ############ compute beta
        if X[tau1].shape[0] > 1:

            inv = inv_sigma(X[tau1])

            ############################## CHECK inv
            # s = sigma(X[tau1])
            # check = cp.einsum('ijk, ikl -> ijl' , s, inv)
            # id = cp.repeat(cp.eye(d).reshape(1,d,d), N, axis=0)[tau1]
            # if not cp.allclose(check,id):
            #     print('error')
            #########################
            
            beta = cp.einsum('ijk, ikl -> ijl', inv, J[tau1])

            ############################## CHECK einsum
            # b_check = cp.zeros((N,d,d))[tau1]
            # J_tmp = J[tau1]
            # for j in range(int(sum(tau1))): 
            #     b_check[j] = cp.matmul(inv[j], J_tmp[j])
            # if not cp.allclose(beta, b_check): 
            #     print('error')
            #######################

            ############################## compute H
            tmp = cp.einsum('ijk, ik -> ij ', beta, n**(-0.5) * DW[tau1])

            ############################## CHECK einsum
            # H_c = cp.zeros((N,d))[tau1]
            # idx_tmp = cp.arange(N)[tau1]
            # for j in ridx_tmp: 
            #     H_c[j] = cp.matmul(beta[j], n**(-0.5) * DW[j])
            # if not cp.allclose(cp.einsum('ijk, ik -> ij ', beta, n**(-0.5) * DW[tau1]), H_c): 
            #     print('error')
            #######################

            H[tau1] += tmp / dist.reshape(-1,1) * alpha.reshape(-1,1)

            ############################## compute J

            ############################## CHECK einsum
            # cp.einsum('ijk,ikl->ijl', Dmu(X[tau1]), J[tau1])
            # Dm = Dmu(X)
            # Dm_check = cp.zeros((N,d,d))
            # for j in range(N): 
            #     Dm_check[j] = cp.matmul(Dm[j],J[j])
            # if not cp.allclose(Dm_check, cp.einsum('ijk,ikl->ijl', Dmu(X), J)): 
            #     print('error')
            #######################

            ############################## CHECK einsum
            # Ds = Dsigma(X)
            # Ds_check = cp.zeros((N,d,d))
            # for j in range(N): 
            #     tmp = cp.matmul(Ds[j,0],J[j])*DW[j,0]
            #     tmp += cp.matmul(Ds[j,1],J[j])*DW[j,1]
            #     tmp += cp.matmul(Ds[j,2],J[j])*DW[j,2]
            #     Ds_check[j] = tmp
            # tmp = cp.einsum('ijkl,iln->ijkn', Dsigma(X), J)
            # tmptmp = cp.repeat(cp.eye(d).reshape(1,d,d), N*3, axis=0).reshape(N,3,3,3)
            # tmptmp = cp.einsum('ijkl, ij -> ijkl', tmptmp, DW)
            # tmp = cp.multiply(tmp, tmptmp).sum(axis=1)
            # if not cp.allclose(Ds_check, tmp): 
            #     print('error')
            #######################

            N_star = cp.int(cp.sum(tau1))
            tmp = cp.einsum('ijkl,iln->ijkn', Dsigma(X[tau1]), J[tau1])
            tmptmp = cp.repeat(cp.eye(d).reshape(1,d,d), N_star*3, axis=0).reshape(N_star,3,3,3)
            tmptmp = cp.einsum('ijkl, ij -> ijkl', tmptmp, DW[tau1])
            tmp = cp.multiply(tmp, tmptmp).sum(axis=1)

            J[tau1] += n**(-1) * cp.einsum('ijk,ikl->ijl', Dmu(X[tau1]), J[tau1]) + n**(-0.5) * tmp

        if FIG: 
            X_cpu = cp.asnumpy(X)
            tau_cpu = cp.asnumpy(tau)
            pointsX_i = ax.scatter3D(X_cpu[tau_cpu,0], X_cpu[tau_cpu,1], X_cpu[tau_cpu,2], color="g")
            pointsX_o = ax.scatter3D(X_cpu[tau_cpu==False,0], X_cpu[tau_cpu==False,1], X_cpu[tau_cpu==False,2], color="r")
            plt.pause(1e-7) 
            pointsX_i.remove()
            pointsX_o.remove()
        
        ############################## compute X
        if cp.sum(tau)==0: break
        X[tau] += n**(-1) * mu(X[tau]) + n**(-0.5) * cp.einsum('ijk,ik->ij', sigma(X[tau]), DW[tau])

        ############################## CHECK einsum
        # s = sigma(X)
        # s_check = cp.zeros((N,d))
        # for j in range(N): s_check[j] = cp.matmul(s[j],DW_t_cp[j])
        # if not cp.allclose(s_check,cp.einsum('ijk,ik->ij', sigma(X), DW_t_cp)): 
        #     print('error')
        #######################

    ############################## check if X is inside the sphere  
    dist = cp.square(X[tau]-sphere_c).sum(axis=1)
    idx_tmp =  dist > sphere_r**2
    tmp = cp.arange(N)[tau]
    tau[tmp[idx_tmp]] = False
    tau1[tmp[idx_tmp]] = False

    # theoretically we could be faster memorizing the DW and run J after a "full" run of X
    
    p = cp.sum(tau) / N
    grad = cp.sum(H[tau], axis=0) / N

    return p, grad

def simul_gpu_anti(x_0, N, n, d, 
                sphere_c, sphere_r, 
                mu, sigma, Dmu, Dsigma, inv_sigma, 
                FIG=True):
    
    assert d==len(sphere_c), "The dim. of X and the dim. of the sphere does not coincide"
    
    # cp.random.seed(int(seed))
    # cp.random.randn(3)

    if FIG: 
        plt.ion()
        plt.title('Greek for safety')
        # fig = plt.figure()
        ax = plt.axes(projection='3d')
        # plt.xlabel('X_1')
        # plt.ylabel('X_2')
        # plt.yzlabel('X_3')

        # DRAW SPHERE
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = np.cos(u)*np.sin(v)*cp.asnumpy(sphere_r)
        y = np.sin(u)*np.sin(v)*cp.asnumpy(sphere_r)
        z = np.cos(v)*cp.asnumpy(sphere_r)
        ax.plot_wireframe(x, y, z, color="r")

        # DRAW SQUARE
        # square = [[x1, x2, x3],[X1, X2, X3]]
        # square_cpu = cp.asnumpy(square)
        # for s, e in combinations(cp.array(list(product(square[:,0], square[:,1], square[:,2]))), 2):
        #     if cp.sum(s==e) > 1 : ax.plot3D(*zip(s, e), color="b")

        # plt.show(block=True) 
        plt.pause(1e-7) 
        
    X = cp.repeat(x_0.reshape(1,3), 2*N, axis=0)
    J = cp.repeat(cp.eye(d).reshape(1,d,d), 2*N, axis=0)
    int_tau1 = cp.zeros(2*N)
    H = cp.zeros((2*N,d))

    tau = cp.ones(2*N, dtype='bool')      # true if always inside the square
    tau1 = cp.ones(2*N, dtype='bool')     # true if t<tau1
    
    for i in range(n):
        
        if cp.sum(tau)==0: break

        ############ check if X is inside the square
        # idx_tmp = cp.logical_and(X[tau]-square[0], X[tau]<square[1])
        # idx_tmp = cp.sum(idx_tmp, 1) < 3

        ############ check if X is inside the sphere  
        dist = cp.square(X[tau]-sphere_c).sum(axis=1)
        idx_tmp =  dist > sphere_r**2
        tmp = cp.arange(2*N)[tau]
        tau[tmp[idx_tmp]] = False
        tau1[tmp[idx_tmp]] = False

        ############ check tau1
        tmp = cp.arange(2*N)[tau1]
        tmp = tmp[int_tau1[tau1]>=1.]
        tau1[tmp] = False

        ############ compute distance square
        # dist = cp.concatenate((cp.square(X[tau1]-square[0]), cp.square(X[tau1]-square[1])), axis=1)
        # dist = cp.min(dist, axis=1)         # dist is already squared

        ############ compute distance sphere
        dist = cp.square(X[tau1]-sphere_c).sum(axis=1)
        dist = cp.square(cp.sqrt(dist) - sphere_r)    # dist is already squared

        ############ compute tau1 integral
        N_star = cp.int(cp.sum(tau1))
        int_tau1[tau1] += 1 / (dist * n)
        alpha = cp.ones(N_star)
        if cp.sum(tau1)!=0:
            if cp.max(int_tau1[tau1])>1:
                tmp = cp.arange(N_star)[int_tau1[tau1]>1]
                tmp1 = cp.arange(2*N)[tau1]
                tmp1 = tmp1[tmp]
                alpha[tmp] = dist[tmp] * n * (1-(int_tau1[tmp1]-1/(dist[tmp]*n)))
                int_tau1[tmp1] = int_tau1[tmp1] - (1- alpha[tmp])/ (dist[tmp] * n)

        # if cp.max(int_tau1)>1.+1e-8: print("check int_tau1")
        
        N_star = cp.int(cp.sum(tau))
        DW = cp.random.randn(N,d)
        DW = cp.concatenate((DW,-DW),axis=0)
        DW[cp.logical_not(tau)] = 0.

        ############ compute beta
        if X[tau1].shape[0] > 1:

            inv = inv_sigma(X[tau1])

            ############################## CHECK inv
            # s = sigma(X[tau1])
            # check = cp.einsum('ijk, ikl -> ijl' , s, inv)
            # id = cp.repeat(cp.eye(d).reshape(1,d,d), N, axis=0)[tau1]
            # if not cp.allclose(check,id):
            #     print('error')
            #########################
            
            beta = cp.einsum('ijk, ikl -> ijl', inv, J[tau1])

            ############################## CHECK einsum
            # b_check = cp.zeros((N,d,d))[tau1]
            # J_tmp = J[tau1]
            # for j in range(int(sum(tau1))): 
            #     b_check[j] = cp.matmul(inv[j], J_tmp[j])
            # if not cp.allclose(beta, b_check): 
            #     print('error')
            #######################

            # compute H
            tmp = cp.einsum('ijk, ik -> ij ', beta, n**(-0.5) * DW[tau1])

            ############################## CHECK einsum
            # H_c = cp.zeros((N,d))[tau1]
            # idx_tmp = cp.arange(N)[tau1]
            # for j in ridx_tmp: 
            #     H_c[j] = cp.matmul(beta[j], n**(-0.5) * DW[j])
            # if not cp.allclose(cp.einsum('ijk, ik -> ij ', beta, n**(-0.5) * DW[tau1]), H_c): 
            #     print('error')
            #######################

            H[tau1] += tmp / dist.reshape(-1,1) * alpha.reshape(-1,1)

            # compute J

            ################################ CHECK einsum
            # cp.einsum('ijk,ikl->ijl', Dmu(X[tau1]), J[tau1])
            # 
            # Dm = Dmu(X)
            # Dm_check = cp.zeros((N,d,d))
            # for j in range(N): 
            #     Dm_check[j] = cp.matmul(Dm[j],J[j])
            # if not cp.allclose(Dm_check, cp.einsum('ijk,ikl->ijl', Dmu(X), J)): 
            #     print('error')
            #######################

            ############################## CHECK einsum
            # Ds = Dsigma(X)
            # Ds_check = cp.zeros((N,d,d))
            # for j in range(N): 
            #     tmp = cp.matmul(Ds[j,0],J[j])*DW[j,0]
            #     tmp += cp.matmul(Ds[j,1],J[j])*DW[j,1]
            #     tmp += cp.matmul(Ds[j,2],J[j])*DW[j,2]
            #     Ds_check[j] = tmp
            # tmp = cp.einsum('ijkl,iln->ijkn', Dsigma(X), J)
            # tmptmp = cp.repeat(cp.eye(d).reshape(1,d,d), N*3, axis=0).reshape(N,3,3,3)
            # tmptmp = cp.einsum('ijkl, ij -> ijkl', tmptmp, DW_t_cp)
            # tmp = cp.multiply(tmp, tmptmp).sum(axis=1)
            # if not cp.allclose(Ds_check, tmp): 
            #     print('error')
            #######################

            N_star = cp.int(cp.sum(tau1))
            tmp = cp.einsum('ijkl,iln->ijkn', Dsigma(X[tau1]), J[tau1])
            tmptmp = cp.repeat(cp.eye(d).reshape(1,d,d), N_star*3, axis=0).reshape(N_star,3,3,3)
            tmptmp = cp.einsum('ijkl, ij -> ijkl', tmptmp, DW[tau1])
            tmp = cp.multiply(tmp, tmptmp).sum(axis=1)

            J[tau1] += n**(-1) * cp.einsum('ijk,ikl->ijl', Dmu(X[tau1]), J[tau1]) + n**(-0.5) * tmp

        if FIG: 
            X_cpu = cp.asnumpy(X)
            tau_cpu = cp.asnumpy(tau)
            pointsX_i = ax.scatter3D(X_cpu[tau_cpu,0], X_cpu[tau_cpu,1], X_cpu[tau_cpu,2], color="g")
            pointsX_o = ax.scatter3D(X_cpu[tau_cpu==False,0], X_cpu[tau_cpu==False,1], X_cpu[tau_cpu==False,2], color="r")
            plt.pause(1e-7) 
            pointsX_i.remove()
            pointsX_o.remove()
        
        # compute X
        if cp.sum(tau)==0: break
        X[tau] += n**(-1) * mu(X[tau]) + n**(-0.5) * cp.einsum('ijk,ik->ij', sigma(X[tau]), DW[tau])

        ################################ CHECK einsum
        # s = sigma(X)
        # s_check = cp.zeros((N,d))
        # for j in range(N): s_check[j] = cp.matmul(s[j],DW_t_cp[j])
        # if not cp.allclose(s_check,cp.einsum('ijk,ik->ij', sigma(X), DW_t_cp)): 
        #     print('error')
        #######################

    ############ check if X is inside the sphere  
    dist = cp.square(X[tau]-sphere_c).sum(axis=1)
    idx_tmp =  dist > sphere_r**2
    tmp = cp.arange(2*N)[tau]
    tau[tmp[idx_tmp]] = False
    tau1[tmp[idx_tmp]] = False

    # theoretically we could be faster memorizing the DW and run J after a "full" run of X
    
    p = cp.sum(tau) / (2*N)
    grad = cp.sum(H[tau], axis=0) / (2*N)

    return p, grad

def mu_gpu(x):
    # N, d = x.shape
    ############ 1
    # return cp.array([x[:,0], x[:,1]**2/1000, x[:,2]]).T
    ############ 2
    return cp.array([x[:,0], 0.5*(x[:,0]+x[:,1]), 1/3*(x[:,0]+x[:,1]+x[:,2])]).T
    ############ 3
    # return x
    ############ 4
    # return cp.cos(x)

def Dmu_gpu(x): 
    
    N, d = x.shape
    ############ 1
    # tmp = cp.array([cp.ones(N), 2*x[:,1]/1000, cp.ones(N)]).T
    # return cp.einsum('ij,jk->ijk', tmp, cp.eye(d))
    ############ 2
    return cp.array([[[1.,0.,0.],[0.5,0.5,0.],[1/3,1/3,1/3]]]).repeat(N, axis=0)
    ############ 3
    # return cp.repeat(cp.eye(d).reshape(1,d,d), N, axis=0)
    ############ 4
    # return cp.einsum('ij,jk->ijk', -cp.sin(x), cp.eye(d))

def sigma_gpu(x, rho=0.5):
    
    N,d = x.shape
    tmp_1 = 2*(1-rho)**0.5 + (1+2*rho)**0.5
    tmp_2 = -(1-rho)**0.5 + (1+2*rho)**0.5
    
    sigma = 1/3 * cp.array([[tmp_1, tmp_2, tmp_2], 
                            [tmp_2, tmp_1, tmp_2], 
                            [tmp_2, tmp_2, tmp_1]])

    # tmp = cp.einsum('ij,jk->ijk', x, cp.eye(3))
    # return cp.einsum('ijk,kl->ijl', tmp, sigma)
    
    return sigma.reshape(1,d,d).repeat(N,axis=0)

def Dsigma_gpu(x, rho=0.5):
    
    N, d = x.shape
    return cp.zeros((N,d,d,d))
    
    # tmp_1 = 2*(1-rho)**0.5 + (1+2*rho)**0.5
    # tmp_2 = -(1-rho)**0.5 + (1+2*rho)**0.5
    
    # Dsigma = 1/3 * cp.array([[tmp_1, tmp_2, tmp_2], 
    #                         [tmp_2, tmp_1, tmp_2], 
    #                         [tmp_2, tmp_2, tmp_1]])
    
    # tmp = cp.concatenate((cp.diag(Dsigma[0,:]).reshape(1,3,3), 
    #                     cp.diag(Dsigma[1,:]).reshape(1,3,3),
    #                     cp.diag(Dsigma[2,:]).reshape(1,3,3)),
    #                     axis=0)#.reshape(x.shape[0],3,3)

    
    # tmp = tmp.reshape(1,-1).repeat(N,axis=0).reshape(N,d,d,d)
    # return tmp
    # return cp.einsum('ijkl,il->ijkl', tmp, x)

def inv_sigma_gpu(x, rho=0.5):
    N,d = x.shape
    tmp_1 = 2/(1-rho)**0.5 + 1/(1+2*rho)**0.5
    tmp_2 = -1/(1-rho)**0.5 + 1/(1+2*rho)**0.5
    
    invsigma = 1/3 * cp.array([[tmp_1, tmp_2, tmp_2], 
                            [tmp_2, tmp_1, tmp_2], 
                            [tmp_2, tmp_2, tmp_1]])
    
    # invx = cp.array([1/(x[:,0]**2/100+10), 1/(x[:,1]**2/100+10), 1/(x[:,2]**2/100+10)]).T
    # invx = cp.array([1/x[:,0], 1/x[:,1], 1/x[:,2]]).T
    # invx = cp.einsum('ij,jk -> ijk', invx, cp.eye(3))
    
    # return cp.einsum('ij,kjl -> kil', invsigma, invx)
    return invsigma.reshape(1,d,d).repeat(N,axis=0)

def minimize(x_0, N, n, d, iter_max,
                sphere_c, sphere_r, 
                mu, sigma, Dmu, Dsigma, inv_sigma,
                constr,
                p_treshold, lr, method="ADAM",
                FIG_minimize=False,
                FIG_simul=False ):

    p = cp.array([2.])
    beta_1 = cp.array([0.9])
    beta_2 = cp.array([0.999])
    m, v = cp.array([0., 0., 0.]), cp.array([0., 0., 0.])
    step = cp.array([1])
    err = cp.array([2.])
    
    x_tm1 = cp.copy(x_0)
    x = cp.copy(x_0)
    
    outside = cp.square(x - sphere_c).sum() > sphere_r**2
    assert cp.logical_not(outside), "ERROR: x_0 is outside the region, change starting point"

    if FIG_minimize: 
        plt.ion()
        plt.title('Greek for safety')
        # fig = plt.figure()
        ax = plt.axes(projection='3d')
        # plt.xlabel('X_1')
        # plt.ylabel('X_2')
        # plt.yzlabel('X_3')

        # DRAW SPHERE
        u_sphere, v_sphere = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_sphere = np.cos(u_sphere)*np.sin(v_sphere)*cp.asnumpy(sphere_r)
        y_sphere = np.sin(u_sphere)*np.sin(v_sphere)*cp.asnumpy(sphere_r)
        z_sphere = np.cos(v_sphere)*sphere_r
        ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="g")

        # DRAW SQUARE
        # square = [[x1, x2, x3],[X1, X2, X3]]
        # square_cpu = np.asnumpy(square)
        # for s, e in combinations(np.array(list(product(square[:,0], square[:,1], square[:,2]))), 2):
        #     if np.sum(s==e) > 1 : ax.plot3D(*zip(s, e), color="b")

        # plt.show(block=True) 
        plt.pause(1e-7) 

    iter = 1

    tmp = cp.concatenate((x,p,cp.zeros(d)),axis=0)
    points = cp.array([tmp])
    err_min = cp.array([1e-2])
    while err>err_min:

        ptm1 = cp.copy(p)
    
        p, grad_p_tau = simul_gpu_anti(x, N, n, d,
                                sphere_c, sphere_r, 
                                mu, sigma, Dmu, Dsigma, inv_sigma,
                                FIG_simul)
        tmp = cp.concatenate((x,cp.array([p]),grad_p_tau),axis=0).reshape(1,-1)
        points = cp.concatenate((points, tmp), axis=0)
        err_signed = p-p_treshold
        if cp.abs(err_signed)>err: print("Error Increased")

        err = cp.abs(err_signed)
        print('p = ', p, '| err = ', err, '| x = ', x)
        
        x_tm1 = cp.copy(x)
        
        grad = err_signed*grad_p_tau
        
        if method == "GD":
            x -= lr*grad
        elif method == "mom":  
            m = beta_1*m + (1-beta_1)*grad
            x -= lr * m
        elif method == "ADAM":  
            m = beta_1*m + (1-beta_1)*grad
            v = beta_2*v + (1-beta_2)*cp.power(grad,2)
            m_hat = m/(1-beta_1**step)
            v_hat = v/(1-beta_2**step)
            x -= lr*m_hat/(cp.sqrt(v_hat)+1e-8)
        else:
            print("Optimization method not recognised")
            return cp.nan, cp.nan, cp.nan

        if constr != None:
            if 'eq' in constr.keys():
                tmp = constr['eq'].astype(bool)
                x[tmp] = constr['eq_value']
            if 'ineq' in constr.keys():
                if cp.dot(constr['ineq'], x) <= constr['ineq_value']:
                    x -= cp.dot(constr['ineq'], x-x_0) / cp.dot(constr['ineq'], constr['ineq']) * constr['ineq']
        
        if FIG_minimize: 
            pointsX = ax.scatter3D(cp.asnumpy(x[0]), cp.asnumpy(x[1]), cp.asnumpy(x[2]), color="r", s=0.1)
            plt.pause(1e-7) 
            # pointsX.remove()
        
        ############ check if x is inside the sphere  
        outside = cp.square(x - sphere_c).sum() > sphere_r**2
        if outside:   
            x = cp.copy(x_tm1)
            print("The algo went outside the region, we go back one step in time")

            # reset param for the optimization part
            m, v = cp.array([0., 0., 0.]), cp.array([0., 0., 0.])
            step = cp.array([1])
        
        step += 1
        iter += 1
        if iter == iter_max:
            print("Not completely minimized, out of iterations")
            break
    
    print("finished minimize")
    return x_tm1, grad_p_tau, p, points[1:]

def parabola(x,y):
    n = x.shape[0]
    x_new = cp.concatenate((cp.square(x), x, cp.ones(n))).reshape(-1,n).T
    coeff = cp.linalg.inv(x_new.T @ x_new) @ x_new.T @ y
    res = cp.square(x_new @ coeff - y).sum()
    # numpy
    # p_q, res_q = np.polyfit(x.get(), y.get(), 2, full=True)[:2]
    return coeff, res 

if __name__ == '__main__':

    x_0 = cp.array([21., 20., 19.])
    # x_0 = cp.array([21.88793009, 20.88690687, 19.89320027])
    N = 10000
    n = 200
    iter_max = 5e1
    d = 3
    # square = [[x1, x2, x3],[X1, X2, X3]]
    sphere_c = cp.array([0., 0., 0.])
    sphere_r = cp.array([100.])
    p_treshold = cp.array([0.5])
    
    sphere_c_cpu = cp.asnumpy(sphere_c)
    sphere_r_cpu = cp.asnumpy(sphere_r)
    p_treshold_cpu = cp.asnumpy(p_treshold)
    
    p = cp.array([2.])
    FIG = False
    
    method = "ADAM"
    if method=="ADAM": lr = cp.array([5e-2])
    if method=="mom": lr = cp.array([5e-2])
    if method=="GD": lr = cp.array([5e-2])
    
    constr = None
    
    # DW = np.random.randn(N_max, n_max, d)

    x, grad_tau, p, points_total = minimize(x_0, N, n, d, iter_max,
                                        sphere_c, sphere_r, 
                                        mu_gpu, sigma_gpu, Dmu_gpu, Dsigma_gpu, inv_sigma_gpu,
                                        constr,
                                        p_treshold, lr, method,
                                        False)
    err_min = cp.array([1e-2])
    while cp.abs(p-p_treshold)>err_min:
        print('optimal p far away')
        x, grad_tau, p, tmp = minimize(x, N, n, d, iter_max,
                            sphere_c, sphere_r, 
                            mu_gpu, sigma_gpu, Dmu_gpu, Dsigma_gpu, inv_sigma_gpu,
                            constr, 
                            p_treshold, lr, method,
                            False)
        points_total = cp.concatenate((points_total, tmp), axis=0)

    plane_z = cp.copy(x[-1])
    print("Point found")
    print('x = ', x, ' | grad = ', grad_tau, ' | p = ', p, ' | plane_z = ', plane_z)
    step_exploration = cp.array([1.5])

    plt.ion()
    # ax = plt.axes(projection='3d')
    plt.figure(figsize=(10,10))
    plt.title('Safety Region')
    plt.xlabel('X_1')
    plt.ylabel('X_2')
    ax = plt.axes()
    ax.scatter(cp.asnumpy(x[0]), cp.asnumpy(x[1]), marker='x', color="r", s=5.)
    ax.axis([-100, 100, -100, 100])
    # ax.axis([10, 30, 50, 65])
    plt.gca().add_patch(plt.Circle((0, 0), (sphere_r[0]**2-plane_z**2)**0.5, fill=False, color='g'))
    plt.pause(1e-7)

    points = cp.copy(x.reshape(1,-1))
    direc_gp_list, dist_gp_list = cp.array([[]]), cp.array([[]])
    direc_pol_list, dist_pol_list = cp.array([[]]), cp.array([[]])
    iteration = cp.array(1)
    while True:

        direction = cp.array((grad_tau[1], -grad_tau[0], cp.array(0.)))
        direction = direction/cp.linalg.norm(direction)*step_exploration
        
        if direc_gp_list.shape[1] <= cp.array(0): direc_gp_list = direction.reshape(1,-1)
        else: direc_gp_list = cp.concatenate((direc_gp_list, direction.reshape(1,-1)), axis=0)
        
        if cp.logical_not(cp.allclose(cp.dot(direction, grad_tau), cp.array(0.))):
            print('problem, direction and gradient not ortogonal')
        
        pol_points = cp.array(4)  # >=4
        if points.shape[0]>=pol_points:
            point_curves = cp.array(4)
            tmp = cp.minimum(point_curves, points.shape[0])
            # p_x_q, res_x_q, _, _, _ = cp.polyfit(points[-tmp:,0], points[-tmp:,1], 2, full=True)   # x_2 = f(x_1)
            # p_y_q, res_y_q, _, _, _ = cp.polyfit(points[-tmp:,1], points[-tmp:,0], 2, full=True)   # x_1 = f(x_2)
            p_x_q, res_x_q = parabola(points[-tmp:,0], points[-tmp:,1]) # x_2 = f(x_1)
            p_y_q, res_y_q = parabola(points[-tmp:,1], points[-tmp:,0]) # x_1 = f(x_2)
            versus = 2*points[-1,:2] - points[-2,:2]
            res = cp.array((res_x_q, res_y_q))
            if cp.min(res) == res[0]: 
                tmp = points[-1,1] + (2*p_x_q[0]*points[-1,0]+p_x_q[1])*(versus[0]-points[-1,0])
                tmp += p_x_q[0]*(versus[0]-points[-1,0])**2
                tmp = cp.array([versus[0], tmp, plane_z])-points[-1,:]
            elif cp.min(res) == res[1]: 
                tmp = points[-1,0] + (2*p_y_q[0]*points[-1,1]+p_y_q[1])*(versus[1]-points[-1,1])
                tmp += p_y_q[0]*(versus[1]-points[-1,1])**2
                tmp = cp.array([tmp, versus[1], plane_z])-points[-1,:]
                # tmp[0], tmp[1] = tmp[1], tmp[0]
            tmp = tmp / cp.linalg.norm(tmp) * cp.linalg.norm(direction)
            # direc_pol_list.append(tmp)

            if direc_pol_list.shape[1] <= cp.array(0): direc_pol_list = tmp.reshape(1,-1)
            else: direc_pol_list = cp.concatenate((direc_pol_list, tmp.reshape(1,-1)), axis=0)

            if points.shape[0]>pol_points:
                tmp1 = cp.minimum(point_curves, points.shape[0]) - pol_points
                denom = cp.sum(dist_pol_list[-tmp1:])**(2) + cp.sum(dist_gp_list[-tmp1:])**(2)
                alpha_gp = cp.sum(dist_pol_list[-tmp1:])**(2) / denom
                alpha_pol = 1 - alpha_gp
                direction = alpha_gp * direction + alpha_pol * tmp
            else:
                direction = 0.5 * (direction + tmp)

        constr_matrix = [direction, cp.array((0., 0., 1))]   #>=
        ineq_value = cp.dot(direction, x+direction)

        # NUMBER OF POINTS TO BE CHECKED BACKWARD
        n_P = cp.array(5)
        points_tm1_in_direction = cp.dot(direction, points[-n_P:,:].T)
        points_tm1_in_direction = cp.any(points_tm1_in_direction >= ineq_value)
        if points_tm1_in_direction:
            direction = -direction
            constr_matrix[0] = direction   #>=
            ineq_value = cp.dot(direction, x+direction)

            points_tm1_in_direction = cp.dot(direction, points[-n_P:,:].T)
            points_tm1_in_direction = cp.any(points_tm1_in_direction >= ineq_value)
            if points_tm1_in_direction:
                direction = -direction
                constr_matrix[0] = direction   #>=
                ineq_value = cp.dot(direction, x_tm1_0+2**(-step_fix)*direction)

        # constr = scipy.optimize.LinearConstraint(constr_matrix, lb, up)
        constr = {'ineq': constr_matrix[0],
                    'eq': constr_matrix[1],
                    'ineq_value': ineq_value, 
                    'eq_value': plane_z}

        tmp1 = cp.asnumpy(cp.linalg.norm(grad_tau[:2])/step_exploration)[0]
        x_cpu = cp.asnumpy(cp.copy(x))
        arrow_gr = ax.arrow(x_cpu[0], x_cpu[1], #1,2,
                            10*cp.asnumpy(grad_tau[0])/tmp1, 10*cp.asnumpy(grad_tau[1])/tmp1, 
                            color='r')  
        arrow_dir = ax.arrow(x_cpu[0], x_cpu[1], 
                            10*cp.asnumpy(direction[0]), 10*cp.asnumpy(direction[1]), 
                            color='g')
        tmp = cp.linspace(-sphere_r, +sphere_r, 5000)
        line = -direction[0]*tmp[:, 0] + ineq_value
        line /= direction[1]
        line = ax.plot(cp.asnumpy(tmp), cp.asnumpy(line))
        plt.pause(1e-7)

        x += direction
        
        # plot_x = ax.scatter(x[0], x[1], marker='x', color="g", s=50.)
        plt.pause(1e-7)

        x_tm1 = cp.copy(x-direction)
        # iter_max = 50
        x, grad_tau, p, tmp = minimize(x, N, n, d,  iter_max,
                                sphere_c, sphere_r, 
                                mu_gpu, sigma_gpu, Dmu_gpu, Dsigma_gpu, inv_sigma_gpu,
                                constr,
                                p_treshold, lr, method,
                                False)
        points_total = cp.concatenate((points_total, tmp), axis=0)
        p_tm1 = cp.copy(p)
        step_fix = cp.array(1.)
        x_tm1_0 = cp.copy(x_tm1)
        while cp.abs(p-p_treshold)>err_min:
            print('optimal p far away')
            # iter_max = 20
            # x = np.copy(x_tm1)
            
            tmp1 = 2*(x-x_tm1_0)-direction
            tmp1 = tmp1/cp.linalg.norm(tmp1)*step_exploration
            # direction = (direction + tmp1)/2
            direction = tmp1
            line[0].remove()
            arrow_dir.remove()

            ineq_value = cp.dot(direction, x_tm1_0+2**(-step_fix)*direction) 
            
            points_tm1_in_direction = cp.dot(direction, points[-n_P:,:].T)
            points_tm1_in_direction = cp.any(points_tm1_in_direction >= ineq_value)
            if points_tm1_in_direction:
                direction = -direction
                constr_matrix[0] = direction   #>=
                ineq_value = cp.dot(direction, x_tm1_0+2**(-step_fix)*direction)

                points_tm1_in_direction = cp.dot(direction, points[-n_P:,:].T)
                points_tm1_in_direction = cp.any(points_tm1_in_direction >= ineq_value)
                if points_tm1_in_direction:
                    direction = -direction
                    constr_matrix[0] = direction   #>=
                    ineq_value = cp.dot(direction, x_tm1_0+2**(-step_fix)*direction)

            constr_matrix = [direction, cp.array((0., 0., 1))]   #>=
             
            arrow_dir = ax.arrow(x_cpu[0], x_cpu[1], 
                            10*cp.asnumpy(direction[0]), 10*cp.asnumpy(direction[1]), 
                            color='g')
            tmp = cp.linspace(-sphere_r, +sphere_r, 5000)
            line = -direction[0]*tmp[:, 0] + ineq_value
            line /= direction[1]
            line = ax.plot(cp.asnumpy(tmp), cp.asnumpy(line))
            plt.pause(1e-7)

            x = x_tm1_0 + 2**(-step_fix)*direction
            
            constr = {'ineq': constr_matrix[0],
                        'eq': constr_matrix[1],
                        'ineq_value': ineq_value, 
                        'eq_value': plane_z}
            x, grad_tau, p, tmp = minimize(x, N, n, d,  iter_max,
                                sphere_c, sphere_r, 
                                mu_gpu, sigma_gpu, Dmu_gpu, Dsigma_gpu, inv_sigma_gpu,
                                constr,
                                p_treshold, lr, method,
                                False)
            points_total = cp.concatenate((points_total, tmp), axis=0)
            step_fix += 1.
        
        points = cp.concatenate((points, x.reshape(1,-1)), axis=0)
        x_cpu = cp.asnumpy(x)
        ax.scatter(x_cpu[0], x_cpu[1], color="b", s=0.5)
        plt.pause(1e-7)

        arrow_gr.remove()
        arrow_dir.remove()
        line[0].remove()
        
        # dist_gp_list.append(np.linalg.norm(x-x_tm1_0-direc_gp_list[-1]))
        
        if dist_gp_list.shape[1] <= cp.array(0): 
            dist_gp_list = cp.linalg.norm(x-x_tm1_0-direc_gp_list[-1]).reshape(1,-1)
        else: 
            tmp = cp.linalg.norm(x-x_tm1_0-direc_gp_list[-1]).reshape(1,-1)
            dist_gp_list = cp.concatenate((dist_gp_list, tmp), axis=0)
            
            if points.shape[0]==pol_points+1:
                dist_pol_list = cp.linalg.norm(x-x_tm1_0-direc_pol_list[-1]).reshape(1,-1)
            elif points.shape[0]>pol_points+1:
                tmp = cp.linalg.norm(x-x_tm1_0-direc_pol_list[-1]).reshape(1,-1)
                dist_pol_list = cp.concatenate((dist_pol_list, tmp), axis=0)

        iteration += 1
        minim_iter = cp.array(10)
        if iteration>=minim_iter+1:
            if cp.min(cp.linalg.norm(x-points[:-minim_iter],axis=1))<step_exploration:
                break
    np.save('points_total_sphere.npy', points_total.get())
    np.save('points_sphere.npy', points.get())
        