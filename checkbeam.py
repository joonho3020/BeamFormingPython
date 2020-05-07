from BeamForming import *
PI = 3.14159265358979
I = 1j
num = 64

def main():
	phi = PI / 4
	a = 3
	b = 3
	phase_label = - k_ * a / num * phi
	Ex_test, Ey_test = aperture_field_uniform(0, phase_label, a, b, num)

	D_test = get_D(Ex_test, Ey_test, - Ey_test/etha, Ex_test/etha, a, b, k_, num)
	D_test[D_test<0] = 0
	#print(D_test)
	#print(np.max(D_test))
	x0 = a / 2
	y0 = b / 2
	d = 5
	Ex_source, Ey_source, Hx_source, Hy_source = aperture_field_source(x0, y0, d, a, b, k_, num)
	D_source = get_D(Ex_source, Ey_source, Hx_source, Hy_source, a, b, k_, num)
	D_source[D_source<0] = 0
	#print(D_source)
	#print(np.max(D_source))
	plot(D_test)
	plot(D_source)
	plt.show()
	
if __name__ == '__main__':
	main()