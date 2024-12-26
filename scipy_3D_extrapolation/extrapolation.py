#!/usr/bin/env python3


import numpy as np
from matplotlib import pyplot as plt

x = [1200,  2400,   4800,   9600]   # Bunch length um
y = [7.028, 4.764,  3.025,  1.785]  # Wakefield V/pC 





def exp_fit():
    #HAS SOME TROUBLE WITH FITTING. COULDN'T FIGURE IT OUT.
    #assuming exponential
    from scipy.optimize import curve_fit

    def exponential_fit(X, a, b, c):
        return a*np.exp(-b*X) + c

    fitting_parameters, covariance = curve_fit(exponential_fit, x, y, method='lm')
    a, b, c = fitting_parameters
    print (fitting_parameters)

    next_x = 6
    next_y = exponential_fit(next_x, a, b, c)
    print("Solution for x=6: ", next_y)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)

    #plot extrapolated function
    xcont = range(6,10000)
    print(xcont, exponential_fit(xcont,a,b,c))
    ax.semilogx(xcont, exponential_fit(xcont,a,b,c))


    #plot values
    x_plot = x
    x_plot.append(next_x)
    
    y_plot = y
    y_plot.append(next_y)
    
    ax.plot(x_plot, y_plot, 'o')

    #show plot
    # plt.show()

    #save plot
    plt.savefig('version1.png', fomat='png')

    plt.close()

    return







def poly2fit():
    #assuming third degree polynomial
    from scipy.optimize import curve_fit

    def poly2(X, a, b, c):
        return a + b*X + c* X**2         

    fitting_parameters, covariance = curve_fit(poly2, x, y)
    a, b, c = fitting_parameters

    next_x = 6
    next_y = poly2(next_x, a, b, c)


    print("Fitting a 2nd degree polynome.")
    print("Solution for x=6: ", next_y)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)

    #plot extrapolated function
    xcont = np.linspace(6,10000,1000)
    ax.plot(xcont, poly2(xcont,a,b,c))


    #plot values
    x_plot = x[:]
    x_plot.append(next_x)
    
    y_plot = y[:]
    y_plot.append(next_y)
    
    ax.plot(x_plot, y_plot, 'o')
    ax.set_title('Polynom 2nd degree fit')

    #show plot
    # plt.show()

    #save plot
    plt.savefig('polynom-2dg.png', fomat='png')

    plt.close()


    return











def poly3fit():
    #assuming third degree polynomial
    from scipy.optimize import curve_fit

    def poly3(X, a, b, c, d):
        return a + b*X + c* X**2 + d* X**3
        

    fitting_parameters, covariance = curve_fit(poly3, x, y)
    a, b, c, d = fitting_parameters

    next_x = 6
    next_y = poly3(next_x, a, b, c, d)

    print("Fitting a 3rd degree polynome.")
    print("Solution for x=6: ", next_y)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)

    #plot extrapolated function
    xcont = np.linspace(6,10000,1000)
    ax.plot(xcont, poly3(xcont,a,b,c,d))


    #plot values
    x_plot = x[:]
    x_plot.append(next_x)
    
    y_plot = y[:]
    y_plot.append(next_y)
    
    ax.plot(x_plot, y_plot, 'o')
    ax.set_title('Polynom 3rd degree fit')

    #show plot
    # plt.show()

    #save plot
    plt.savefig('polynom-3dg.png', fomat='png')

    plt.close()


    return








def poly3logfit():
    #assuming third degree polynomial
    from scipy.optimize import curve_fit

    def poly3(X, a, b, c, d):
        X = np.log(X)
        return a + b*X + c* X**2 + d* X**3
        

    fitting_parameters, covariance = curve_fit(poly3, x, y)
    a, b, c, d = fitting_parameters

    next_x = 6
    next_y = poly3(next_x, a, b, c, d)

    print("Fitting a 3rd degree polynome for log(x).")
    print("Solution for x=6: ", next_y)

    fig = plt.figure(figsize=(10,6))
    ax = fig.add_subplot(1,1,1)

    #plot extrapolated function
    xcont = np.linspace(6,10000,1000)
    ax.plot(np.log(xcont), poly3(xcont, a, b , c, d))


    #plot values
    x_plot = x[:]
    x_plot.append(next_x)
    
    y_plot = y[:]
    y_plot.append(next_y)
    
    ax.plot(np.log(x_plot), y_plot, 'o')
    ax.set_title('Polynom 3rd degree fit')
    ax.set_xlabel('log x')

    #show plot
    # plt.show()

    #save plot
    plt.savefig('polynom-3dg-logx.png', fomat='png')

    plt.close()


    return









# def poly4fit():
    # TOO FEW DATA POINTS; CAN'T HANDLE IT
    # #assuming third degree polynomial
    # from scipy.optimize import curve_fit
    # 
    # def poly4(X, a, b, c, d, e):
    #     return a + b*X + c* X**2 + d* X**3 + e* X**4
    #     
    # 
    # fitting_parameters, covariance = curve_fit(poly4, x, y)
    # a, b, c, d, e = fitting_parameters
    # 
    # next_x = 6
    # next_y = poly4(next_x, a, b, c, d, e)
    # 
    # print("Fitting a 4th degree polynome.")
    # print("Solution for x=6: ", next_y)
    # 
    # fig = plt.figure(figsize=(10,6))
    # ax = fig.add_subplot(1,1,1)
    # 
    # #plot extrapolated function
    # xcont = np.linspace(6,10000,1000)
    # ax.plot(xcont, poly4(xcont,a,b,c,d,e))
    # 
    # 
    # #plot values
    # x_plot = x[:]
    # x_plot.append(next_x)
    # 
    # y_plot = y[:]
    # y_plot.append(next_y)
    # 
    # ax.plot(x_plot, y_plot, 'o')
    # ax.set_title('Polynom 4th degree fit')
    # #show plot
    # plt.show()
    # 
    # #save plot
    # plt.savefig('polynom-4dg.png', fomat='png')
    # 
    # plt.close()
    # 
    # 
    # return




if __name__ == "__main__":

    poly2fit()
    poly3fit()
    # exp_fit()
    poly3logfit()




