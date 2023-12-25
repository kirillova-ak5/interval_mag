import numpy as np
from mpmath import *
from twin import *
from intvalpy import *
from operation import *

# Press the green button in the gutter to run the script.
def examp():
    #Interval(float('nan'), float('nan'))
    data1 = Interval(1, 10)
    data2 = Interval(1, 100)
    data3 = Interval(float('nan'), float('nan'))
    data4 = Interval(2, 6)
    T1 = Twin(data1, data2)
    print("T1 = ", T1)
    T2 = Twin(data1, data4)
    print("T2 = ", T2)

    print("T1 + T2 = ", T1 + T2)
    print("T1 * T2 = ", T1 * T2)
    print(-T2)
    print(T1 == T1)
    print(T1 == T2)
    print(T1 in T1)

