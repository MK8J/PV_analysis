import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

class teal_psf():
    def __init__(self,len_lsf_in):
        xx = np.arange(1, len_lsf_in, dtype=np.float)
        lsf = 1.0/xx**2.0
        psf_teal = self.teals_method(lsf)
        fig1 = plt.figure(2)
        graph_psf = plt.plot(psf_teal)
        plt.setp(graph_psf, 'color', 'r', 'linewidth', 1.0)
        plt.ylabel('1D PSF Intensity (arb)')
        #plt.semilogy()
        plt.show()

    def teals_method(self, LSF):
        LSF /= np.amax(LSF)
        coefficient_matrix = self.generate_coefficient_matrix(len(LSF))
        #find outermost psf value
        psf_teal =  np.zeros((len(LSF),1),dtype=float)
        psf_teal[len(LSF)-1] = LSF[len(LSF)-1]/coefficient_matrix[len(LSF)-1,len(LSF)-1]
        #find all psf values
        for counter1 in range(len(LSF)-2,-1,-1):
            psf_teal[counter1] = (LSF[counter1]- np.sum(psf_teal[counter1+1:len(LSF)-1 ]*coefficient_matrix[counter1,(counter1+1):len(LSF)-1])) \
                /(coefficient_matrix[counter1,counter1])      
        return psf_teal

    def generate_coefficient_matrix(self,len_lsf):
        coefficient_matrix = np.zeros((len_lsf, len_lsf),dtype=float)
        for counter1 in range(0,len_lsf):
            x_min = (counter1 - 0.5)
            x_max = (counter1 + 0.5)
            
            for counter2 in range(counter1,len_lsf):
                r_min = (counter2 - 0.5);
                r_max = (counter2 + 0.5);
                coefficient_matrix[counter1,counter2] = 2.0 * self.calculate_enclosed_area(x_min,x_max,r_min,r_max)
                #print coefficient_matrix[counter1,counter2]
        X = np.arange(0, len(coefficient_matrix[1,:]), dtype=np.float)
        Y = np.arange(0, len(coefficient_matrix[:,1]), dtype=np.float)
        X, Y = np.meshgrid(X, Y)
        fig2 = plt.figure(2)
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, coefficient_matrix,rstride=10,cstride=10)
        plt.show()
        return coefficient_matrix
    def calculate_enclosed_area(self,x_min,x_max,r_min,r_max):
        # this function returns the area enclosed by the input bounding area
        # it used the formula:
        # integral (sqrt(r_max^2-x^2) dx - integral (sqrt(r_min^2-x^2) dx
        # the limits of the integral are x_max,and x_min
        # an analytical solution exists for the integral(sqrt(r^2-x^2) dx 
        # = 1/2(x.sqrt(r^2-x^2)+r^2*np.arctan(x/sqrt(r^2-x^2)))
        
        #abc = 0.5*(x_max*np.sqrt(r_max**2-x_max**2)+r_max**2*np.arctan(x_max/np.sqrt(x_max/np.sqrt(r_max**2-x_max**2))))- \
        #    0.5*(x_min*np.sqrt(r_max**2-x_min**2)+r_max**2*np.arctan(x_min/np.sqrt(x_min/np.sqrt(r_max**2-x_min**2))))

        #asc = 0.5*(x_max*np.sqrt(r_min**2-x_max**2)+r_min**2*np.arctan(x_max/np.sqrt(x_max/np.sqrt(r_min**2-x_max**2))))- \
        #    0.5*(x_min*np.sqrt(r_min**2-x_min**2)+r_min**2*np.arctan(x_min/np.sqrt(x_min/np.sqrt(r_min**2-x_min**2))))
        #enclosed_area = abc-asc

        #this if statement prevents divid by zero errors
        if r_max == x_max:
            arctan_expression_max_max = r_max**2*np.pi/2
            sqrt_expression_max_max = 0
        else:
            arctan_expression_max_max = r_max**2*np.arctan(x_max/np.sqrt(r_max**2-x_max**2))
            sqrt_expression_max_max = x_max*np.sqrt(r_max**2-x_max**2)

        if r_max == x_min:
            arctan_expression_max_min = r_max**2*np.pi/2
            sqrt_expression_max_min = 0
        else:
            arctan_expression_max_min = r_max**2*np.arctan(x_min/np.sqrt(r_max**2-x_min**2))
            sqrt_expression_max_min = x_min*np.sqrt(r_max**2-x_min**2)

        if r_min == x_max:
            arctan_expression_min_max = r_min**2*np.pi/2
            sqrt_expression_min_max = 0
        else:
            arctan_expression_min_max = r_min**2*np.arctan(x_max/np.sqrt(r_min**2-x_max**2))
            sqrt_expression_min_max = x_max*np.sqrt(r_min**2-x_max**2)

        if r_min == x_min:
            arctan_expression_min_min = r_min**2*np.pi/2
            sqrt_expression_min_min = 0
        else:
            arctan_expression_min_min = r_min**2*np.arctan(x_min/np.sqrt(r_min**2-x_min**2))
            sqrt_expression_min_min = x_min*np.sqrt(r_min**2-x_min**2)

        area_big_circle = (0.5*(sqrt_expression_max_max + arctan_expression_max_max))- \
            (0.5*(sqrt_expression_max_min + arctan_expression_max_min))
        area_small_circle = (0.5*(sqrt_expression_min_max + arctan_expression_min_max))- \
            (0.5*(sqrt_expression_min_min + arctan_expression_min_min))

        #following if statement also removes r_min less than 0
        if r_min < 0:
            area_small_circle = 0.    
        #following if statement prevents nan error when range of integral does not include positive yvalue of circle    
        if r_min == x_min:   
            area_small_circle = 0.
        if r_max <= x_min:
            enclosed_area = 0.

        enclosed_area = area_big_circle - area_small_circle  
        return enclosed_area
teal_psf(100)
