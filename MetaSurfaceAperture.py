import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as axes3d
import math

PI = 3.14159265358979
wave_length = 1.0
k = 2*PI / wave_length
n_ = 120*PI
num = 64
r = 10 * wave_length
EPS = 1e-3



class Surface():
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.dir = np.zeros((num, num))
    
    """------------------  from here call after calling get_dir ------------------------"""
    
    def get_maxdir(self):
        return np.max(self.dir)
    
    def get_hpbw(self):
        dir_ = self.rotate2origin()
        i, _ = np.unravel_index(np.argmax(dir_, axis=None), dir_.shape)
        hp = self.get_maxdir() - 3
        dir_[dir_<hp] = 0
        t_cnt = np.count_nonzero(dir_[:,0])
        return t_cnt*PI/2/num

    def get_space_angle(self):
        sa = 0
        for i in range(num):
            for j in range(num):
                if self.dir[i, j] >= self.get_maxdir() - 3:
                    sa += np.sin(PI/2/num*i) * (PI**2)/(num**2)
        return sa

    def get_info(self):
        return self.get_maxdir(), self.get_hpbw(), self.get_space_angle()

    def rotate2origin(self):
        i, j = np.unravel_index(np.argmax(self.dir, axis=None), self.dir.shape)
        dir_ = np.copy(self.dir)
        dir_ = np.roll(dir_, -j, axis=1)
        return dir_
    
    def plot_dir(self):
        d = self.dir
        theta = np.linspace(0, PI/2, num)
        phi = np.linspace(0, PI*2, num)
        phi_, theta_ = np.meshgrid(phi, theta)
        x = d * np.vstack(np.sin(theta_)) * np.cos(phi_)
        y = d * np.vstack(np.sin(theta_)) * np.sin(phi_)
        z = d * np.vstack(np.cos(theta_))
        
        dmax = np.max(d)
        i, j = np.unravel_index(np.argmax(d, axis=None), d.shape)

        ax = plt.axes(projection='3d')
        ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        ax.set_title('surface');
        ax.set_xlabel('$X$')
        ax.set_ylabel('$Y$')
        ax.set_xlim([-35, 35])
        ax.set_ylim([-35, 35])
        ax.set_zlim(bottom = 0.0)
    
    def plot_plane_dir(self):
        i, j = np.unravel_index(np.argmax(self.dir, axis=None), self.dir.shape)
        theta = np.linspace(0, PI/2, num)
        phi = np.linspace(0, 2*PI, num)
        
        fig, axes = plt.subplots(nrows=1, ncols=2)

        plt.subplot(121)
        plt.plot(theta, self.dir[:,j])
        plt.xlabel('theta (rad)')
        plt.ylabel('Directivity Theta Plane (dB)')
        
        plt.subplot(122)
        plt.plot(phi, self.dir[i,:])
        plt.xlabel('phi (rad)')
        plt.ylabel('Directivity Phi Plane (dB)')

        fig.tight_layout() # Or equivalently,  "plt.tight_layout()"
        plt.show()


"""
metasurface with continuous E field distribution

nx, ny : number of pixels in x, y direction
dx : size of pixel
"""

class MetaSurfaceAperture(Surface):
    def __init__(self, ax, ay):
        nx = ny = num

        super(MetaSurfaceAperture, self).__init__(nx, ny)
        self.dx = ax/nx
        self.dy = ay/ny
        self.ax = ax
        self.ay = ay
    
    def set_zero_field(self):
        self.ex = np.zeros((self.nx, self.ny), dtype=np.complex128)
        self.ey = np.zeros((self.nx, self.ny), dtype=np.complex128)
        self.hx = np.zeros((self.nx, self.ny), dtype=np.complex128)
        self.hy = np.zeros((self.nx, self.ny), dtype=np.complex128)

    def set_uniform_e_field(self):
        self.set_zero_field()
        self.ex = np.ones((self.nx, self.ny))
    
    def set_horn_e_field(self, d):
        a = 1.0
        b = 1.0
        self.set_zero_field()
        for i in range(self.nx):
            for j in range(self.ny):
                x = self.ax*i/self.nx - self.ax/2
                y = self.ay*j/self.ny - self.ay/2
                r = np.sqrt(x**2 + y**2)
                R = np.sqrt(r**2 + d**2)

                cos_theta = d / R
                sin_theta = r / R

                if r == 0:
                    cos_phi = 1
                    sin_phi = 1
                else:
                    cos_phi = x/r
                    sin_phi = y/r

                coeff = 1j*a*b*k*np.exp(-1j*k*R)/(2*PI*R)
                X = k*a/2*sin_theta*cos_phi
                Y = k*b/2*sin_theta*sin_phi
                if abs(X) < EPS:
                    X = 1
                else:
                    X = np.sin(X)/X
                if abs(Y) < EPS:
                    Y = 1
                else:
                    Y = np.sin(Y)/Y
                F = X*Y
                et = 0.5*coeff*sin_phi*(1+cos_theta)*F
                ep = 0.5*coeff*cos_phi*(1+cos_theta)*F
                ht = -ep/n_
                hp = et/n_
                self.ex[i, j] = et*cos_theta*cos_phi - ep*sin_phi
                self.ey[i, j] = et*cos_theta*sin_phi + ep*cos_phi
                self.hx[i, j] = ht*cos_theta*cos_phi - hp*sin_phi
                self.hy[i, j] = ht*cos_theta*sin_phi + hp*cos_phi

    def set_dipole_e_field(self, d):
        self.set_zero_field()
        coeff = 1j*60

        for i in range(self.nx):
            for j in range(self.ny):
                x = self.ax * i / self.nx - self.ax/2
                y = self.ay * j / self.ny - self.ay/2
                r = np.sqrt(d**2 + y**2)
                R = np.sqrt(r**2 + x**2)
                cos_theta = x / R
                sin_theta = r / R
                cos_phi = y / r
                sin_phi = d / r
                F = np.cos(PI / 2 * cos_theta) / sin_theta;
                AF = (1 + 2 * np.cos(PI * sin_theta * cos_phi)) * (1 + 2 * np.cos(PI * sin_theta * sin_phi));
       
                F = F * AF;
                H_phi = 1/n_ * np.exp(- 1j * k * R) / R  * F;
                E_theta = np.exp(- 1j * k * R) / R * F;
            
                self.hx[i, j] = 0;
                self.hy[i, j] = - coeff * H_phi * sin_phi;
                self.ex[i, j] = - coeff * E_theta * sin_theta;
                self.ey[i, j] = coeff * E_theta * cos_theta * cos_phi;
    
    def set_phase(self, theta0, phi0):
        px = np.sin(theta0)*np.cos(phi0)
        py = np.sin(theta0)*np.sin(phi0)
        xid = np.arange(0, self.nx, 1)
        yid = np.arange(0, self.ny, 1)
        x_expon = np.exp(-1j*k*self.dx*px*xid)
        y_expon = np.exp(-1j*k*self.dy*py*yid)
        self.ex = self.ex*(np.vstack(x_expon)*y_expon)
        self.ey = self.ey*(np.vstack(x_expon)*y_expon)
        self.hx = self.hx*(np.vstack(x_expon)*y_expon)
        self.hy = self.hy*(np.vstack(x_expon)*y_expon)

    def equivalence_theorem(self):
        mx = self.ey
        my = -self.ex
        jx = -self.hy
        jy = self.hx
        return (mx, my, jx, jy)

    def get_far_field(self, theta0, phi0):
        self.set_phase(theta0, phi0)
        mx, my, jx, jy = self.equivalence_theorem()
        n_phi   = np.zeros((num, num), dtype=np.complex128)
        n_theta = np.zeros((num, num), dtype=np.complex128)
        l_phi   = np.zeros((num, num), dtype=np.complex128)
        l_theta = np.zeros((num, num), dtype=np.complex128)

        theta = np.linspace(0, PI/2, num)
        phi = np.linspace(0, 2*PI, num)
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)
        sin_phi = np.sin(phi)
        cos_phi = np.cos(phi)

        for i in range(num):
            for j in range(num):
                exponent = 1j * k * ( (i - num/2)* self.dx * np.vstack(sin_theta) * cos_phi  \
                                    + (j - num/2)* self.dy * np.vstack(sin_theta) * sin_phi)
                exponent = np.exp(exponent) * self.dx * self.dy
                n_theta += (jx[i, j] * np.vstack(cos_theta) * cos_phi + jy[i, j] * np.vstack(cos_theta) * sin_phi) * exponent
                n_phi += (- jx[i, j] * sin_phi + jy[i, j] * cos_phi) * exponent
                l_theta += (mx[i, j] * np.vstack(cos_theta) * cos_phi + my[i, j] * np.vstack(cos_theta) * sin_phi) * exponent
                l_phi += (- mx[i, j] * sin_phi + my[i, j] * cos_phi) * exponent
        return (l_phi + n_theta*n_, -l_theta + n_phi*n_)


    def get_dir(self, theta0, phi0):
        theta = np.linspace(0, PI/2, num)
        sin_theta = np.vstack(np.sin(theta))
        
        e_theta, e_phi = self.get_far_field(theta0, phi0)
        u = np.real(e_theta*np.conj(e_theta) + e_phi*np.conj(e_phi))
        pr = 1/(4*PI)*np.sum(u*sin_theta*(PI*PI/(num**2))) 
        self.dir = 10*np.log10(u/pr)
        self.dir[self.dir<0] = 0