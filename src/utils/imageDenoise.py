import numpy as np

def svt(mat, tau):
    u, s, v = np.linalg.svd(mat, full_matrices = 0)
    vec = s - tau
    vec[vec < 0] = 0
    return np.matmul(np.matmul(u, np.diag(vec)), v)

def LRMC(sparse_mat, dense_mat, rho, maxiter):
    
    pos_train = np.where(sparse_mat != 0)
    pos_test = np.where((sparse_mat == 0) & (dense_mat != 0))
    binary_mat = sparse_mat.copy()
    binary_mat[pos_train] = 1
    
    X = sparse_mat.copy()
    Z = sparse_mat.copy()
    T = sparse_mat.copy()
    rse = np.zeros(maxiter)
    
    for it in range(maxiter):
        Z = svt(X + T / rho, 1 / rho)
        X = Z - T / rho
        X[pos_train] = sparse_mat[pos_train]
        T = T - rho * (Z - X)
        rse[it] = (np.linalg.norm(X[pos_test] - dense_mat[pos_test], 2) 
                   / np.linalg.norm(dense_mat[pos_test], 2))
    return X, rse

#read image
import imageio
import matplotlib.pyplot as plt

lena = imageio.imread('z6.bmp')/255.0
sparse_lena=imageio.imread('z6ss.bmp')/255.0
print('The shape of the image is {}.'.format(lena.shape))

dim1, dim2,dim3 = lena.shape
mask = np.round(np.random.rand(dim1, dim2,dim3))  # Generate a binary mask.
mask1 = np.round(np.random.rand(dim1, dim2,dim3))
# mask2 = np.round(np.random.rand(dim1, dim2))

plt.figure(figsize=(15,12))
plt.imshow(lena)
plt.title('The original Lena')
plt.axis('off')

plt.figure(figsize=(15,12))
plt.imshow(sparse_lena)
plt.title('The incomplete Lena')
plt.axis("off")

plt.show()
#deal
import time

start = time.time()
rho = 0.005
maxiter = 50
mat_hat, rse_svt = LRMC(sparse_lena[:,:,0], lena[:,:,0], rho, maxiter)
print('Running time: %d seconds.'%(end - start))
start = time.time()
mat_hat1, rse_svt1 = LRMC(sparse_lena[:,:,1], lena[:,:,1], rho, maxiter)
print('Running time: %d seconds.'%(end - start))
start = time.time()
mat_hat2, rse_svt2 = LRMC(sparse_lena[:,:,2], lena[:,:,2], rho, maxiter)
print('Running time: %d seconds.'%(end - start))

#修复完把三张图拼接在一起
c=[]
for i in range(dim1):
    c.append([])
    for j in range(dim2):
        c[i].append([mat_hat[i][j],mat_hat1[i][j],mat_hat2[i][j]])
#
plt.figure(figsize=(20,15))
# plt.imshow(mat_hat)
plt.imshow(c)
# plt.imshow(mat_hat2)
plt.savefig("lbk.png")
plt.axis('off')
plt.show()