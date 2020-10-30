import numpy as np
import pandas as pd
import torch as T
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d

# row : theta  column : phi
# row : x      column : y

PI = 3.14159265358979
I = 1j
num = 64

theta = np.array([PI/2/num*i for i in range(num)])
phi = np.array([2*PI/num*i for i in range(num)])
sin_theta = np.sin(theta)
cos_theta = np.cos(theta)
sin_phi = np.sin(phi)
cos_phi = np.cos(phi)

wave_length = 1
k_ = 2*PI/wave_length
etha = 120 * PI

def np_multi(A, B, axis):
    A = np.array(A)
    B = np.array(B)
    
    swapped_shape = A.swapaxes(0, axis).shape

    for dim_step in range(A.ndim-1):
        B = B.repeat(swapped_shape[dim_step+1], axis=0)\
             .reshape(swapped_shape[:dim_step+2])
        
    B = B.swapaxes(0, axis)
    return A * B

def get_propagated_E(Js_x, Js_y, Ms_x, Ms_y, a, b, k_, num):
    N_phi = np.zeros((num, num), dtype=np.complex128)
    N_theta = np.zeros((num, num), dtype=np.complex128)
    L_theta = np.zeros((num, num), dtype=np.complex128)
    L_phi = np.zeros((num, num), dtype=np.complex128)
    
    for i in range(num):
        for j in range(num):
            exponent = I * k_ * ( (i - num/2)/num * a * np.vstack(sin_theta) * cos_phi  \
                                + (j - num/2)/num * b * np.vstack(sin_theta) * sin_phi)
            exponent = np.exp(exponent) * a / num * b / num


            N_phi += (- Js_x[i, j] * sin_phi + Js_y[i, j] * cos_phi) * exponent
            N_theta += (Js_x[i, j] * np.vstack(cos_theta) * cos_phi + Js_y[i, j] * np.vstack(cos_theta) * sin_phi) * exponent
            L_theta += (Ms_x[i, j] * np.vstack(cos_theta) * cos_phi + Ms_y[i, j] * np.vstack(cos_theta) * sin_phi) * exponent
            L_phi += (- Ms_x[i, j] * sin_phi + Ms_y[i, j] * cos_phi) * exponent

    #return E_theta, E_theta
    return (L_phi + N_theta * etha, -L_theta + N_phi * etha)

def get_D(Ex, Ey, Hx, Hy, a, b, k_, num):
    Ms_x = Ey
    Ms_y = - Ex
    Js_x = - Hy
    Js_y = Hx

    (E_theta, E_phi) = get_propagated_E(Js_x, Js_y, Ms_x, Ms_y, a, b, k_, num)

    E_total_square = np.real(E_theta * np.conj(E_theta) + E_phi * np.conj(E_phi))
    P_radiate = np.sum(E_total_square * np.vstack(sin_theta)) * PI * PI / (num ** 2)

    return 10 * np.log10(4 * PI * E_total_square / P_radiate)

def aperture_field_uniform(mode, phi, a, b, num):
    if mode==0:
        Ex = np.ones((num, num))
        Ey = np.zeros((num, num))
    #we just need uniform E field for now...

    phase_angle = np.exp(I * np.transpose(np.reshape(np.array([phi * i for i in range(num)] * num), (num, num))))

    return (Ex * phase_angle, Ey * phase_angle)


def aperture_field_source(x0, y0, d, a, b, k_, num):

    x = np.vstack(np.array([a * i / num for i in range(num)]))
    y = np.array([b * i / num for i in range(num)])
    x_ = x - x0
    y_ = y - y0

    r = np.sqrt(d * d + y_**2)
    R = np.sqrt(r * r + x_**2)

    cos_t = x_ / R
    sin_t = r / R
    cos_p = y_ / r
    sin_p = d / r

    F = np.cos(PI / 2 * cos_t) / sin_t
    exponent = np.exp(-I * k_ * R)

    H_phi = exponent / R * F
    E_theta = H_phi * etha

    return (-E_theta*sin_t, E_theta * cos_t * cos_p, np.zeros((num, num)), -H_phi * sin_p)

def plot(D):
    phi, theta = np.linspace(0, 2 * np.pi, num), np.linspace(0, np.pi / 2, num)
    PHI, THETA = np.meshgrid(phi, theta)
    R = np.cos(PHI**2)
    X = D * np.sin(THETA) * np.cos(PHI)
    Y = D * np.sin(THETA) * np.sin(PHI)
    Z = D * np.cos(THETA)
    
    D_max = np.max(D)
    
    ax = plt.axes(projection='3d')
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title('surface');
    ax.set_xlim([-35, 35])
    ax.set_ylim([-35, 35])
    ax.set_zlim(bottom = 0.0)