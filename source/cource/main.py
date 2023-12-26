import random

import example
import smt
from intvalpy import *
from twin import *
# first - inner, second - outer
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LogNorm

def plotTol(trueAnswer, b_i1, b_i2, xx, x2):
    x_grid_ = np.arange(-5, 5, 0.25)
    y_grid_ = np.arange(-5, 5, 0.25)
    gridsize = len(x_grid_)
    x_grid, y_grid = np.meshgrid(x_grid_, y_grid_)
    b_grid = np.empty(shape=(gridsize, gridsize, 7))
    z_grid = np.empty(shape=(gridsize, gridsize))
    z_grid2 = np.empty(shape=(gridsize, gridsize))
    for i in range(gridsize):
        for j in range(gridsize):
            b_grid[i][j] = np.array([x_grid[i][j], y_grid[i][j], trueAnswer[2], trueAnswer[3], trueAnswer[4], trueAnswer[5], trueAnswer[6]])
            z_grid[i][j] = linear.Tol.value(A, b_i1, b_grid[i][j])
            z_grid2[i][j] = linear.Tol.value(A, b_i2, b_grid[i][j])
    c1 = plt.contour(x_grid, y_grid, z_grid, linestyles='dashed')
    plt.clabel(c1)
    c2 = plt.contour(x_grid, y_grid, z_grid2)
    plt.clabel(c2)
    plt.scatter(xx[0][0], xx[0][1], s=100, label= "Outer interval solution")
    plt.scatter(x2[0][0], x2[0][1], s=30, label= "Inner interval solution")
    plt.scatter(trueAnswer[0], trueAnswer[1], marker='*', label= "Real solution")
    plt.legend()
    plt.figure()

    for i in range(gridsize):
        for j in range(gridsize):
            b_grid[i][j] = np.array([trueAnswer[0], trueAnswer[1], x_grid[i][j], y_grid[i][j], trueAnswer[4], trueAnswer[5], trueAnswer[6]])
            z_grid[i][j] = linear.Tol.value(A, b_i1, b_grid[i][j])
            z_grid2[i][j] = linear.Tol.value(A, b_i2, b_grid[i][j])
    c1 = plt.contour(x_grid, y_grid, z_grid, linestyles='dashed')
    plt.clabel(c1)
    c2 = plt.contour(x_grid, y_grid, z_grid2)
    plt.clabel(c2)
    plt.scatter(xx[0][2], xx[0][3], s=100, label= "Outer interval solution")
    plt.scatter(x2[0][2], x2[0][3], s=30, label= "Inner interval solution")
    plt.scatter(trueAnswer[2], trueAnswer[3], marker='*', label= "Real solution")
    plt.legend()
    plt.figure()

    for i in range(gridsize):
        for j in range(gridsize):
            b_grid[i][j] = np.array([trueAnswer[0], trueAnswer[1], trueAnswer[2], trueAnswer[3], x_grid[i][j], y_grid[i][j], trueAnswer[6]])
            z_grid[i][j] = linear.Tol.value(A, b_i1, b_grid[i][j])
            z_grid2[i][j] = linear.Tol.value(A, b_i2, b_grid[i][j])
    c1 = plt.contour(x_grid, y_grid, z_grid, linestyles='dashed')
    plt.clabel(c1)
    c2 = plt.contour(x_grid, y_grid, z_grid2)
    plt.clabel(c2)
    plt.scatter(xx[0][4], xx[0][5], s=100, label= "Outer interval solution")
    plt.scatter(x2[0][4], x2[0][5], s=30, label= "Inner interval solution")
    plt.scatter(trueAnswer[4], trueAnswer[5], marker='*', label= "Real solution")
    plt.legend()
    plt.figure()

    for i in range(gridsize):
        for j in range(gridsize):
            b_grid[i][j] = np.array([trueAnswer[0], trueAnswer[1], trueAnswer[2], trueAnswer[3], trueAnswer[4], x_grid[i][j], y_grid[i][j]])
            z_grid[i][j] = linear.Tol.value(A, b_i1, b_grid[i][j])
            z_grid2[i][j] = linear.Tol.value(A, b_i2, b_grid[i][j])
    c1 = plt.contour(x_grid, y_grid, z_grid, linestyles='dashed')
    plt.clabel(c1)
    c2 = plt.contour(x_grid, y_grid, z_grid2)
    plt.clabel(c2)
    plt.scatter(xx[0][5], xx[0][6], s=100, label= "Outer interval solution")
    plt.scatter(x2[0][5], x2[0][6], s=30, label= "Inner interval solution")
    plt.scatter(trueAnswer[5], trueAnswer[6], marker='*', label= "Real solution")
    plt.legend()


def bubble_max_row(m, b, col):
    max_element = m[col][col]
    max_row = col
    for i in range(col + 1, len(m)):
        if abs(m[i][col]) > abs(max_element):
            max_element = m[i][col]
            max_row = i
    if max_row != col:
        m[col], m[max_row] = m[max_row], m[col]
        b[col], b[max_row] = b[max_row], b[col]


def solve_gauss(m, b):
    n = len(m)
    n2 = len(m[0])
    # forward trace
    for k in range(n2 - 1):
        bubble_max_row(m, b, k)
        for i in range(k + 1, n):
            div = m[i][k] / m[k][k]
            #ttt = -Twin(Interval(div,div), Interval(div,div))
            ttt = Twin(b[k].X_l * (-div), b[k].X * (-div))
            b[i] = b[i] + ttt
            for j in range(k, n2):
                m[i][j] -= div * m[k][j]

    # check modified system for nonsingularity
    #if is_singular(m):
        #print('The system has infinite number of answers...')
        #return
    # backward trace
    x = [Twin(Interval(0, 0), Interval(0, 0)) for i in range(n2)]
    for k in range(n2 - 1, -1, -1):
        if m[k][k] == 0:
            continue
        d = Twin(Interval(0, 0), Interval(0, 0))
        for j in range(k + 1, n2):
            d = d + Twin(x[j].X_l * m[k][j], x[j].X * m[k][j])
        t = b[k] + (-d)
        x[k] = Twin(t.X_l / m[k][k], t.X / m[k][k])

    # Display results
    return x


def is_singular(m):
    for i in range(len(m)):
        if not m[i][i]:
            return True


def MSE(x, A, b):
    n = len(x)
    m = len(b)
    bx = [0] * m
    rss = 0
    for i in range(m):
        bx[i] = Twin(Interval(0, 0), Interval(0, 0))
        for j in range(n):
            bx[i] +=Twin(A[i][j], A[i][j]) * Twin(Interval(x[j], x[j]), Interval(x[j], x[j]))
        rsstw=((b[i] + (- bx[i])) * (b[i] + (- bx[i])))
        rss += rsstw.X_l.a + rsstw.X_l.b / 2
    return rss


if __name__ == '__main__':
    #example.examp()
    #smt.smtmain()
    xt = [0] * (17-9)
    yt = [0] * (11-7)
    data = [[0]*(11-7) for i in range(9, 17)]
    data2 = [[0]*(11-7) for i in range(9, 17)]
    for pix in range(9, 17):
        for cir in range(7, 11):
            xt[pix-9] = pix
            yt[cir-7] = cir
            smt.NUM_PIXEL = pix
            smt.NUM_CIRCLE = cir
            k, b = smt.init_pixels(smt.NUM_PIXEL)
            A = smt.matrixA(k, b, smt.NUM_CIRCLE)
            Asrc = smt.matrixA(k, b, smt.NUM_CIRCLE)
            start_x = -2
            random.seed(30)
        #    trueAnswer = [1 + random.uniform(-1, 1) for i in range(len(A[0]))]
            trueAnswer = [0.1 + pow(i - 3, 1) for i in range(len(A[0]))]
            trueAnswer[0:3] = [trueAnswer[2 - i] * (0.5) for i in range(3)]
            trueAnswer = trueAnswer[2:len(A[0])] + trueAnswer[0:2]
            b1 = [0] * len(b)
            for i in range(len(A)):
                for j in range(len(A[i])):
                    b1[i] += A[i][j] * trueAnswer[j]
            b = b1
            #print(trueAnswer)
            for i in range(len(A)):
                for j in range(len(A[i])):
                    A[i][j] = np.longdouble(A[i][j])
                    A[i][j] = [A[i][j], A[i][j]]
            A = Interval(A)
            b_i1 = Interval([[b[i] * 0.75, b[i] * 1.25] for i in range(len(b))])
            b_i2 = Interval([[b[i] * 0.9, b[i] * 1.1] for i in range(len(b))])
            b_tw = [Twin(b_i2[i], b_i1[i]) for i in range(len(b))]
            xx = linear.Tol.maximize(A, b_i1)
            x2 = linear.Tol.maximize(A, b_i2)
            tru = trueAnswer
            xx1 = xx[0]
            xx2 = x2[0]
            data[pix - 9][cir-7] = np.linalg.norm(tru-xx1)
            data2[pix - 9][cir-7] = np.linalg.norm(tru-xx2)
            if pix==13 and cir==7:
                print("Outer", data[pix - 9][cir-7])
                print("Inner", data2[pix - 9][cir - 7])
            #x2 = linear.Tol.value(A, b_i1, trueAnswer)

#    print(xx)
#    print(x2)
#    print(trueAnswer)
            x0 = [i for i in range(len(trueAnswer))]
#    for i in range(len(b)):
#        plt.plot((i, i), (b_tw[i].X.a, b_tw[i].X.b), 'r')
#        plt.plot((i, i), (b_tw[i].X_l.a, b_tw[i].X_l.b), 'b')

#    plt.xlabel('x')
#    plt.ylabel('y')
#    plt.grid()
#    plt.title("twin vector")
#    plt.figure()
    #xxx = solve_gauss(Asrc, b_tw)
    #for i in range(len(xxx)):
    #    print(xxx[i])
    fig, ax = plt.subplots()
    for row in range(len(data)):
        for cell in range(len(data[0])):
            data[row][cell] = float(data[row][cell])
            data2[row][cell] = float(data2[row][cell])
    data = np.array(data)
    data2 = np.array(data2)
    #plt.imshow(data, cmap=plt.colormaps["summer"])
    #ax.set_xticks(np.arange(len(yt)), labels=yt)
    #ax.set_yticks(np.arange(len(xt)), labels=xt)
    #plt.colorbar()
    #plotTol(trueAnswer, b_i1, b_i2, xx, x2)
    cm = sns.diverging_palette(110, 180, s=100, as_cmap=True)
    cm=sns.color_palette("crest", as_cmap=True)
    htmap = sns.heatmap(data, cmap=cm, norm=LogNorm(data.min(),data.max()),
            cbar_kws={"ticks":[1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
                               1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0,1,10,1e2]},
            vmin = 1e-13, vmax=100, annot=True)
    htmap.set_xticks(np.arange(len(yt)), labels=yt)
    htmap.set_yticks(np.arange(len(xt)), labels=xt)
    htmap.set(xlabel='Circles', ylabel='Pixels')
    htmap.set(title='Outer intervals')
    plt.figure()
    htmap = sns.heatmap(data2, cmap=cm, norm=LogNorm(data.min(),data.max()),
            cbar_kws={"ticks":[1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7,
                               1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0,1,10,1e2]},
            vmin = 1e-13, vmax=100, annot=True)
    htmap.set_xticks(np.arange(len(yt)), labels=yt)
    htmap.set_yticks(np.arange(len(xt)), labels=xt)
    htmap.set(xlabel='Circles', ylabel='Pixels')
    htmap.set(title='Inner intervals')
    plt.show()

    print("a")


