import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from math import sqrt

NUM_CIRCLE = 7
DIFF_RADIUS = 0.15
NUM_PIXEL = 13
DIST_PIXELS = 0.03

XMIN = -1
XMAX = 4
YMIN = -1
YMAX = 1

def intersection_points(r, k, b):
    get_sqrt = pow(r, 2)*(1+pow(k, 2)) - pow(b, 2) 
    if(get_sqrt < 0):
        return 0
    elif(get_sqrt == 0):
        return 1
    else:
        return 2

def init_pixels(n):
    xt, yt = 3, 0
    x_pixels = 4
    
    if (n == 1):
        k = [0]
        b = [0]
        return k, b

    y, k, b, x = [], [], [], []

    x.append(x_pixels)
    y.append(DIST_PIXELS*(n-1)/2)
    k.append((yt-y[0]) / ( xt - x[0]))
    b.append(y[0] - k[0]*x[0])
    for i in range(1, n):
        x.append(x_pixels)
        y.append(y[i-1]-DIST_PIXELS)
        k.append((yt-y[i]) / ( xt - x[i]))
        b.append(y[i] - k[i]*x[i])
    return k, b

def draw_circles_lines(coef_k, coef_b):
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = 0
    # draw circle
    for i in range(NUM_CIRCLE):
        radius += DIFF_RADIUS
        a = radius * np.cos( theta )
        b = radius * np.sin( theta )
        plt.plot( a , b, 'k')
    # draw lines
    for i in range(len(coef_k)):
        x = np.linspace(XMIN, XMAX, 10)
        y = coef_k[i]*x+coef_b[i]
        plt.plot(x, y, 'r--')


def draw_dots(num_circles, coef_k, coef_b):
    radius = 0
    for i in range(num_circles):
        radius += DIFF_RADIUS
        for j in range(len(coef_k)):
            ip = intersection_points(radius, coef_k[j], coef_b[j])
            if(ip == 0):
                continue
            elif(ip == 1):
                x = (-coef_k[j]*coef_b[j]) / (1+pow(coef_k[j], 2))
                y = coef_k[j]*x + coef_b[j]
                plt.plot(x, y, 'coef_b*')
            else:
                x1 = (-coef_k[j]*coef_b[j] + sqrt(pow(radius, 2)*(1+pow(coef_k[j], 2)) - pow(coef_b[j], 2) )) / ((1+pow(coef_k[j], 2)))
                x2 = (-coef_k[j]*coef_b[j] - sqrt(pow(radius, 2)*(1+pow(coef_k[j], 2)) - pow(coef_b[j], 2) )) / ((1+pow(coef_k[j], 2)))
                y1 = coef_k[j]*x1 + coef_b[j]
                y2 = coef_k[j]*x2 + coef_b[j]
                plt.plot(x1, y1, 'b*')
                plt.plot(x2, y2, 'b*')


def len_c(a, b, c, d):
    return sqrt(abs(a-b)**2 + abs(c-d)**2)

def matrixA(coef_k, coef_b, m):
    prevD = 0
    D = 0
    n = len(coef_k) #count of pixel

    A = [ [0]*m for i in range(n) ]
    for i in range(n):
        radius = 0
        prevD = 0
        for j in range(m):
            radius += DIFF_RADIUS
            ip = intersection_points(radius, coef_k[i], coef_b[i])
            if(ip == 0):
                continue
            elif(ip == 1):
                A[i][j] = 0
                prevD = D = 0
            else:
                x1 = (-coef_k[i]*coef_b[i] + sqrt(pow(radius, 2)*(1+pow(coef_k[i], 2)) - pow(coef_b[i], 2) )) / ((1+pow(coef_k[i], 2)))
                x2 = (-coef_k[i]*coef_b[i] - sqrt(pow(radius, 2)*(1+pow(coef_k[i], 2)) - pow(coef_b[i], 2) )) / ((1+pow(coef_k[i], 2)))
                y1 = coef_k[i]*x1 + coef_b[i]
                y2 = coef_k[i]*x2 + coef_b[i]
                D = len_c(x1,x2,y1,y2)
                if prevD == 0:
                    A[i][j] = D
                    prevD = D
                else:
                    A[i][j] = abs(D - prevD)
                    prevD = D


    # for i in range(len(coef_k)):
    #     for j in range(m):
    #         print('{:2.3f}'.format(A[i][j]), end=" ") 
    #     print()
            
    return A

def plot_one_g():
    plt.figure(figsize=[15, 6])
    k, b = init_pixels(NUM_PIXEL)
    draw_circles_lines(k, b)
    draw_dots(NUM_CIRCLE, k, b)
    A = matrixA(k, b, NUM_CIRCLE)
    R = np.linalg.cond(A)
    #displaying the condition number of the matrix.
    print('the condition number of the matrix is = {}'.format(R))
    plt.xlim(XMIN, XMAX)
    plt.ylim(YMIN, YMAX)
    plt.grid()
    plt.figure()
    return A

def plot_different_count_pixels():
    num = 15
    DIST_PIXELS = 0.01
    for i in range(7):
        DIST_PIXELS += 0.01
        matrix_cond = []
        x = []
        for i in range(10, num):
            x.append(i)
            k, b = init_pixels(i)
            A = matrixA(k, b, NUM_CIRCLE)
            matrix_cond.append(np.linalg.cond(A))
        plt.figure(figsize=[4, 4])
        plt.grid()
        plt.semilogy(x, matrix_cond, '--or')
        plt.title("{:.2f}".format(DIST_PIXELS) + "  расстояние между пикселями")
    


def smtmain():
    A = plot_one_g()
    for i in range(len(A)):
        for j in range(len(A[0])):
            print("{:.4f}".format(A[i][j]), end = ' ')
        print("")
        
    # plot_different_count_pixels()


