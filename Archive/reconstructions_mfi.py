

import numpy as np
import matplotlib.pyplot as plt
from skimage.draw import ellipse

# -------------------------------------------------------------------------

fname = 'projections.npy'
print('Reading:', fname)
projections = np.load(fname)

print('projections:', projections.shape, projections.dtype)

# -------------------------------------------------------------------------

fname = 'signals_data.npy'
print('Reading:', fname)
signals_data = np.load(fname)

print('signals_data:', signals_data.shape, signals_data.dtype)

fname = 'signals_time.npy'
print('Reading:', fname)
signals_time = np.load(fname)

print('signals_time:', signals_time.shape, signals_time.dtype)

# -------------------------------------------------------------------------

P = projections.reshape((projections.shape[0], -1))

print('P:', P.shape, P.dtype)

# -------------------------------------------------------------------------

n_rows = projections.shape[1]
n_cols = projections.shape[2]

Dh = np.eye(n_rows*n_cols) - np.roll(np.eye(n_rows*n_cols), 1, axis=1)
Dv = np.eye(n_rows*n_cols) - np.roll(np.eye(n_rows*n_cols), n_cols, axis=1)

print('Dh:', Dh.shape, Dh.dtype)
print('Dv:', Dv.shape, Dv.dtype)

# -------------------------------------------------------------------------

ii, jj = ellipse(n_rows//2, n_cols//2, n_rows//2, n_cols//2)
mask = np.ones((n_rows, n_cols))
mask[ii,jj] = 0.

Io = np.eye(n_rows*n_cols) * mask.flatten()

print('Io:', Io.shape, Io.dtype)

# -------------------------------------------------------------------------

Pt = np.transpose(P)
PtP = np.dot(Pt, P)

DtDh = np.dot(np.transpose(Dh), Dh)
DtDv = np.dot(np.transpose(Dv), Dv)
ItIo = np.dot(np.transpose(Io), Io)

alpha_1 = .01# np.identity(n_cols*n_rows)
alpha_2 = .01# np.identity(n_cols*n_rows)
alpha_3 = alpha_1*1000


print(np.mean(PtP)*PtP.shape[0]*PtP.shape[1])

for f in signals_data.transpose()[3:]:
    print(f.shape)
    inv = np.linalg.inv(PtP + alpha_1*DtDh + alpha_2*DtDv + alpha_3*ItIo)
    error = np.inf
    while np.abs(error-1) > .1:
        M = np.dot(inv, Pt)
        g_mfi = np.dot(M,f)
        f_v = np.dot(P,g_mfi)
        error = np.dot(f_v-f,f_v-f)/(32*0.0018**2)
        print(error)
        if np.abs(error-1) <= .1:
            print('converged')
            plt.imshow(g_mfi.reshape((n_rows,n_cols)),extent = [-10., 10., -10., 10.])
            plt.show()
            break
        else:
            diag = 1./np.maximum(g_mfi,1e-8)
            W = np.diag(diag)
            inv = np.linalg.inv(PtP + alpha_1*np.dot(Dh.transpose(),np.dot(W,Dh)) + alpha_2*np.dot(Dv.transpose(),np.dot(W,Dv)) + alpha_1*1e3*ItIo)
            # plt.imshow(g_mfi.reshape((n_rows,n_cols)),vmin = 0, vmax = 0.001, extent = [-100., 100., -100., 100.])
            # print(np.max(g_mfi))
            # plt.show()
        

#     inv = np.linalg.inv(PtP + alpha_1*DtDh + alpha_2*DtDv + alpha_3*ItIo)
#     
#     g_mfi = np.dot(M,f)

# # -------------------------------------------------------------------------

# tomo = []
# tomo_t = np.arange(0., signals_time[0,-1], 0.01)

# for t in tomo_t:
#     i = np.argmin(np.fabs(signals_time[0] - t))
#     f = signals_data[:,i].reshape((-1, 1))
#     g = np.dot(M, f)
#     tomo.append(g.reshape((n_rows, n_cols)))
#     print(signals_time[0,i])

# tomo = np.array(tomo)

# print('tomo:', tomo.shape, tomo.dtype)
# print('tomo_t:', tomo_t.shape, tomo_t.dtype)

# # -------------------------------------------------------------------------

# vmin = 0.
# vmax = np.max(tomo)

# ni = 4
# nj = tomo.shape[0]/ni
# nj = 6

# fig, ax = plt.subplots(ni, nj, figsize=(2*nj, 2*ni))


# for i in range(ni):
#     for j in range(nj):
#         k = i*nj + j
#         ax[i,j].imshow(tomo[k], vmin=vmin, vmax=vmax)
#         ax[i,j].set_title('t=%.3fs' % tomo_t[k])
#         ax[i,j].set_axis_off()

# plt.show()

# # -------------------------------------------------------------------------


# f_v = np.dot(P,np.clip(tomo.reshape(tomo.shape[0],-1).transpose(),a_min = 0, a_max = None))
# f = signals_data
# print('f_v:', f_v.shape)
# print('f:', f.shape)

# print(f[8,8])
# print(f_v[8,8])
# print(np.dot(P[8],tomo[8].flatten()))


# plt.figure()
# plt.imshow(P[8].reshape(n_rows,n_cols),extent = [-100., 100., -100., 100.])
# plt.colorbar()
# plt.figure()
# print(tomo[8].shape)
# plt.imshow(tomo[8],extent = [-100., 100., -100., 100.])
# plt.colorbar()
# plt.show()

# for i in range(f.shape[0]):
# 	plt.figure()
# 	plt.plot(tomo_t,f_v[i],'r')
# 	plt.plot(tomo_t,f[i],'b')
# 	plt.show()