from mpmath import *
from intvalpy import Interval
from intvalpy import intersection
#import operation as op

def phi(I1, I2):
    Z = intersection(I1, I2)

    if not (isnan(float(Z.a)) or isnan(float(Z.b))):
        return Interval(float('nan'), float('nan'))

    min_ = abs(I1.a - I2.a)
    min_a = I1.a
    min_b = I2.a

    if abs(I1.b - I2.b) < min_:
        min_ = abs(I1.b - I2.b)
        min_a = I1.b
        min_b = I2.b

    if abs(I1.a - I2.b) < min_:
        min_ = abs(I1.a - I2.b)
        min_a = I1.a
        min_b = I2.b

    if abs(I1.b - I2.a) < min_:
        min_ = abs(I1.b - I2.a)
        min_a = I1.b
        min_b = I2.a

    return Interval(min_a, min_b)


def psi(I1, I2):
    return Interval(
        min(I1.a, I1.b, I2.a, I2.b),
        max(I1.a, I1.b, I2.a, I2.b)
    )

class Twin(object):
    def __init__(self, X_l, X):
        # I1 and I2 must be a single intervals. If it have property "len", there are not single intervals
        if hasattr(X_l, 'len') or hasattr(X, 'len'):
            print("I1 or I2 is not single intervals. Check your data.")
            exit()

        if isnan(float(X.a)) or isnan(float(X.b)):
            print("The outer interval cannot be empty.")
            exit()

        self.X_l = X_l

        if (isnan(float(X_l.a)) or isnan(float(X_l.b))) and X.a == X.b:
            self.X_l = X

        self.X = X

        if isnan(self.X_l.a):
            self.X_l_width = -1
        else:
            self.X_l_width = self.X_l.wid

        self.X_width = self.X.wid

    def __str__(self):
        return "[" + str(self.X_l) + ", " + str(self.X) + "]"

    def p(self, other):
        if self.X_l_width == -1 and other.X_l_width == -1:
            return float('nan')

        if self.X_l_width == -1:
            return other.X_l.a + self.X.b

        if other.X_l_width == -1:
            return self.X_l.a + other.X.b

        return min(self.X_l.a + other.X.b, other.X_l.a + self.X.b)

    def q(self, other):
        if self.X_l_width == -1 and other.X_l_width == -1:
            return float('nan')

        if self.X_l_width == -1:
            return other.X_l.b + self.X.a

        if other.X_l_width == -1:
            return self.X_l.b + other.X.a

        return max(self.X_l.b + other.X.a, other.X_l.b + self.X.a)

    def __add__(self, other):
        if self.X_width <= other.X_l_width or other.X_width <= self.X_l_width:
            return type(other)(
                Interval(self.p(other), self.q(other)),
                Interval(self.X.a + other.X.a, self.X.b + other.X.b)
            )
        else:
            return type(other)(
                Interval(float('nan'), float('nan')),
                Interval(self.X.a + other.X.a, self.X.b + other.X.b)
            )

    def __mul__(self, other):
        if self.X_l_width == -1 and other.X_l_width == -1:
            return Twin(
                Interval(float('nan'), float('nan')),
                Interval(self.X.a, self.X.b) * Interval(other.X.a, other.X.b)
            )

        if self.X_l_width == -1:
            return Twin(
                phi(
                    other.X_l.a * Interval(self.X.a, self.X.b),
                    other.X_l.b * Interval(self.X.a, self.X.b)
                ),
                Interval(self.X.a, self.X.b) * Interval(other.X.a, other.X.b)
            )

        if other.X_l_width == -1:
            return Twin(
                phi(
                    self.X_l.a * Interval(other.X.a, other.X.b),
                    self.X_l.b * Interval(other.X.a, other.X.b)
                ),
                Interval(self.X.a, self.X.b) * Interval(other.X.a, other.X.b)
            )

        return Twin(
            psi(
                phi(
                    (self.X_l.a) * Interval(other.X.a, other.X.b),
                    (self.X_l.b) * Interval(other.X.a, other.X.b)
                ),
                phi(
                    other.X_l.a * Interval(self.X.a, self.X.b),
                    other.X_l.b * Interval(self.X.a, self.X.b)
                )
            ),
            Interval(self.X.a, self.X.b) * Interval(other.X.a, other.X.b))

    def __neg__(self):
        return Twin(-self.X_l, -self.X)

    def __invert__(self):
        if 0 in self.X_l or 0 in self.X:
            print("ERROR:Cannot be divided into intervals containing 0.")
            exit()

        return Twin(1 / self.X_l, 1 / self.X)

    def __eq__(self,other):
        if self.X_l.a == other.X_l.a and self.X_l.b == other.X_l.b and self.X.a == other.X.a and self.X.b == other.X.b:
            return True
        else:
            return False

    def __contains__(self, other):
        if (self.X in other.X) and (other.X_l in self.X_l):
            return True
        else:
            return False
