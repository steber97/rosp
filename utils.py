from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt

EPS = 1e-5

def zero_one_with_probability(prob = 0.5):
    if np.random.rand() <= prob:
        return 1
    return 0


def create_rand_matrix(n, prob=0.5):
    M = np.identity(n)
    for i in range(n):
        M[i,i] = n-2
    for i in range(n):
        for j in range(i+1, n):
            M[i,j] = 2 * (-0.5 + zero_one_with_probability(prob))
            M[j,i] = M[i,j]
    return M

def diagonally_dominant(M):
    return np.all(
        (2*np.sum((M * np.eye(M.shape[0])), axis=0) - np.sum(np.abs(M), axis=0)) >= 0
    )

def psd(M):
    return np.linalg.eig(M)[0].min() >= 0

def is_vect_eigenv(M: np.array, v: np.array):
    """
    Check if v is an eigenvector for M
    """
    w = M @ v
    w = w / v
    w -= w[0]
    return np.all(np.abs(w) < 1e-6)

class XYPoint:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return "Point ({:.2f}, {:.2f})".format(self.x, self.y)
class PiecewiseSegment:
    def __init__(self, p1: XYPoint, p2: XYPoint, m: float):
        """
        If p1 or p2 are None, it means that the piecewise segment
        goes to or from +- infinity.
        At least one of them needs to be not none.
        
        :param p1: begin of segment.
        :param p2: end of segment.
        :param m: angular coefficient.
        """
        self.p1 = p1
        self.p2 = p2
        assert not (p1 is None and p2 is None)
        self.m = m
        if self.p1 is not None and self.p2 is not None:
            assert p1.x <= p2.x
            if  abs((p2.x - p1.x) * m + p1.y - p2.y) > EPS:
                print(self)
                raise AssertionError

    def __repr__(self):
        return "{}, {}, m = {:.2f}".format(self.p1, self.p2, self.m)
    
    def min(self, ps: PiecewiseSegment) -> List[PiecewiseSegment]:
        """
        Given two piecewise segments, return the minimum of them. 
        This can be a piecewise function made of only one piecewise segment 
        (in case one of the two segments lies below the other)
        or a piecewise function of two segments (in case they cross at some point in the interval).
        Notice that the two segments should be defined over the same interval p1.x->p2.x. 
        
        :param ps: other piecewise segment
        :return: A piecewise function (with at most 2 segments) encoding min(self, ps).
        """
        assert ((self.p1 is None and ps.p1 is None) or abs(self.p1.x - ps.p1.x) < EPS) and ((self.p2 is None and ps.p2 is None) or abs(self.p2.x - ps.p2.x) < EPS)
        segments = []
        if self.p1 is None and ps.p1 is None:
            assert self.p2 is not None and ps.p2 is not None
            assert abs(self.p2.x - ps.p2.x) < EPS
            if abs(self.m - ps.m) < EPS:
                if (self.p2.y <= ps.p2.y):
                    segments.append(self)
                else:
                    segments.append(ps)
            else:
                x = (-ps.m * ps.p2.x + self.m * self.p2.x - self.p2.y + ps.p2.y) / (self.m - ps.m)
                y = self.m * (x - self.p2.x) + self.p2.y
                if x >= self.p2.x or abs(x - self.p2.x) < EPS:
                    if self.m > ps.m:
                        segments.append(self)
                    else:
                        segments.append(ps)
                else:
                    if self.m > ps.m:
                        segments.append(PiecewiseSegment(None, XYPoint(x, y), self.m))
                        segments.append(PiecewiseSegment(XYPoint(x, y), ps.p2, ps.m))
                    else:
                        segments.append(PiecewiseSegment(None, XYPoint(x, y), ps.m))
                        segments.append(PiecewiseSegment(XYPoint(x, y), self.p2, self.m))


        elif self.p2 is None and ps.p2 is None:
            assert self.p1 is not None and ps.p1 is not None
            assert abs(self.p1.x - ps.p1.x) < EPS
            if abs(self.m - ps.m) < EPS:
                if self.p1.y <= ps.p1.y:
                    segments.append(self)
                else:
                    segments.append(ps)
            else:
                x = (-ps.m * ps.p1.x + self.m * self.p1.x - self.p1.y + ps.p1.y) / (self.m - ps.m)
                y = self.m * (x - self.p1.x) + self.p1.y
                if x <= self.p1.x:
                    if self.m < ps.m:
                        segments.append(self)
                    else:
                        segments.append(ps)
                else:
                    if self.m < ps.m:
                        segments.append(PiecewiseSegment(ps.p1, XYPoint(x, y), ps.m))
                        segments.append(PiecewiseSegment(XYPoint(x,y), None, self.m))
                    else:
                        segments.append(PiecewiseSegment(self.p1, XYPoint(x, y), self.m))
                        segments.append(PiecewiseSegment(XYPoint(x, y), None, ps.m))
        else:
            assert self.p1 is not None and self.p2 is not None and ps.p1 is not None and ps.p2 is not None
            assert abs(self.p1.x - ps.p1.x) < EPS and abs(self.p2.x - ps.p2.x) < EPS
            if ((self.p1.y < ps.p1.y) or abs(self.p1.y - ps.p1.y) < EPS) and (
                (self.p2.y < ps.p2.y) or abs(self.p2.y - ps.p2.y) < EPS):
                segments.append(self)
            elif ((self.p1.y > ps.p1.y) or abs(self.p1.y - ps.p1.y) < EPS) and (
                (self.p2.y > ps.p2.y) or abs(self.p2.y - ps.p2.y) < EPS):
                segments.append(ps)
            elif (self.p1.y < ps.p1.y) and (self.p2.y > ps.p2.y):
                x = (-ps.m * ps.p1.x + self.m * self.p1.x - self.p1.y + ps.p1.y) / (self.m - ps.m)
                assert self.p1.x < x < self.p2.x
                newpoint = XYPoint(x, self.m * x - self.m * self.p1.x + self.p1.y)
                segments.append(PiecewiseSegment(self.p1, newpoint, self.m))
                segments.append(PiecewiseSegment(newpoint, ps.p2, ps.m)) 
            else:
                assert (self.p1.y > ps.p1.y) and (self.p2.y < ps.p2.y)
                x = (-ps.m * ps.p1.x + self.m * self.p1.x - self.p1.y + ps.p1.y) / (self.m - ps.m)
                assert self.p1.x < self.p2.x
                newpoint = XYPoint(x, self.m * x - self.m * self.p1.x + self.p1.y)
                segments.append(PiecewiseSegment(ps.p1, newpoint, ps.m))
                segments.append(PiecewiseSegment(newpoint, self.p2, self.m))
        return segments

class PiecewiseFunction:
    """
    Simple representation as consecutive piecewise segments.
    """

    def __init__(self, segments: List[PiecewiseSegment]):
        self.segments: List[PiecewiseSegment] = segments
        assert len(segments) >= 2
        assert segments[0].p1 is None
        assert segments[-1].p2 is None 
        assert self.check_continuous()
        assert self.check_concave()

    def check_continuous(self) -> bool:
        """
        Check that the piecewise function is continuous.
        Namely, the joints need to coincide.
        """
        for i in range(len(self.segments) - 1):
            if self.segments[i].p2 != self.segments[i+1].p1:
                return False
        return True

    def check_convex(self) -> bool:
        """
        Check that the piecewise linear function is convex: namely,
        that the consecutive angular coefficients are non decreasing.
        """
        if not self.check_continuous():
            return False
        for i in range(len(self.segments) - 1):
            if not(self.segments[i].m <= self.segments[i+1].m or abs(self.segments[i].m - self.segments[i+1].m) < 1e-6):
                return False
        return True

    def check_concave(self) -> bool:
        """
        Check that the piecewise linear function is concave: namely,
        that the consecutive angular coefficients are non increasing.
        """
        if not self.check_continuous():
            return False
        for i in range(len(self.segments) - 1):
            if not(self.segments[i].m >= self.segments[i+1].m or abs(self.segments[i].m - self.segments[i+1].m) < 1e-6):
                return False
        return True

    def shorten(self) -> None:
        """
        If there are consecutive piecewise linear segments with the same angular coefficient,
        merge them into a unique segment.
        The operation happens inplace (self.segments is changed).
        """
        new_segments: List[PiecewiseSegment] = []
        for i in range(len(self.segments)):
            if len(new_segments) == 0:
                new_segments.append(self.segments[i])
            elif abs(new_segments[-1].m - self.segments[i].m) < 1e-6:
                new_segments[-1].p2 = self.segments[i].p2
            else:
                new_segments.append(self.segments[i])
        self.segments = new_segments
    
    def plot(self) -> None:
        avg_len = np.mean([s.p2.x - s.p1.x for s in self.segments[1:-1]])
        xs = []
        ys = []
        for i in range(len(self.segments)):
            seg: PiecewiseSegment = self.segments[i]
            if i == 0:
                assert self.segments[0].p1 is None
                xs.append(seg.p2.x - avg_len)
                ys.append(seg.m * (xs[0] - seg.p2.x) + seg.p2.y)
            if i == len(self.segments)-1:
                assert self.segments[-1].p2 is None
                xs.append(seg.p1.x + avg_len)
                ys.append(seg.m * (xs[-1] - seg.p1.x) + seg.p1.y)
            else:
                xs.append(seg.p2.x)
                ys.append(seg.p2.y)
        plt.plot(xs, ys)

def create_piecewise_linear(m: np.array, v: np.array, i: int) -> PiecewiseFunction:
    """
    Given the coefficients of the function f_i(x) = m[i] - x * v[i] - sum(|m[j] - x * v[j]|),
    return it as a list of points, angular coefficients where there are the non-differentiable
    points.
    
    :param m: Description
    :type m: np.array
    :param v: Description
    :type v: np.array
    :return: Description
    :rtype: List[Tuple[Tuple[float, float], float]]
    """

    return None

def y(point: XYPoint, x: float, m: float) -> float:
    return m * (x - point.x) + point.y

def merge_piecewise_linear(l1: PiecewiseFunction, l2: PiecewiseFunction) -> PiecewiseFunction:
    """
    Given 2 piecewise linear functions, defined as point, angular coefficient,
    return a new piecewise linear function that is the min of the two.
    It is enough to split the two functions according to the same x's,
    and then understand which of the 2 functions is below in the aligned intervals.
    If they cross, add a new point.
    At the end, it would be good to remove useless intervals where the angular coefficient doesn't change.
    
    :param l1: Description
    :param l2: Description
    :return: Description
    """

    def split_piecewise_function(l: PiecewiseFunction, xs: List[float]) -> List[PiecewiseSegment]:
        segments = [PiecewiseSegment(None, XYPoint(xs[0], y(l.segments[0].p2, xs[0], l.segments[0].m)), l.segments[0].m)]
        i = 1 if abs(xs[0] - l.segments[0].p2.x) < EPS else 0
        for j, x in enumerate(xs[:-1]):
            pt1, pt2, m = l.segments[i].p1, l.segments[i].p2, l.segments[i].m
            assert pt1 is None or pt1.x <= x or abs(pt1.x - x) < EPS
            if not(pt2 is None or pt2.x >= xs[j+1] or abs(pt2.x - xs[j+1]) < EPS):
                raise AssertionError
            pt = pt1 if pt1 is not None else pt2
            segments.append(PiecewiseSegment(XYPoint(x, y(pt, x, m)), XYPoint(xs[j+1], y(pt, xs[j+1], m)), m))
            if pt2 is not None and abs(xs[j+1] - pt2.x) < EPS:
                i += 1
        segments.append(PiecewiseSegment(XYPoint(xs[-1], y(l.segments[-1].p1, xs[-1], l.segments[-1].m)), None, l.segments[-1].m))
        return segments

    xs = []
    i = 1
    j = 1
    while i < len(l1.segments) and j < len(l2.segments):
        if abs(l1.segments[i].p1.x - l2.segments[j].p1.x) < EPS:
            xs.append(l1.segments[i].p1.x)
            i += 1
            j += 1
        elif l1.segments[i].p1.x < l2.segments[j].p1.x:
            xs.append(l1.segments[i].p1.x)
            i += 1
        else:
            xs.append(l2.segments[j].p1.x)
            j += 1
        
    xs += [s.p1.x for s in l1.segments[i:]]
    xs += [s.p1.x for s in l2.segments[j:]]

    seg1 = split_piecewise_function(l1, xs)
    seg2 = split_piecewise_function(l2, xs)

    segments = []
    for s1, s2 in zip(seg1, seg2):
        segments += s1.min(s2)

    return PiecewiseFunction(segments)

def binary_search_max(l: PiecewiseFunction) -> float:
    """
    Given a list of points, angular coefficients of a convex function,
    find the x that maximizes it. It is sufficient to look right as long as the angular coefficient is increasing,
    and to look left when the angular coefficient is decreasing.
    
    :param l: Description
    :type l: List[Tuple[Tuple[float, float], float]]
    :return: Description
    :rtype: float
    """
    assert l.check_concave()
    return 0.

def maximize_x(M: np.array, V: np.array) -> float:
    merged_pf = [[]] # merged piecewise functions
    for i in range(len(M)):
        merged_pf[0].append(create_piecewise_linear(M[i,:], V[i,:], i))
    while len(merged_pf[-1]) > 1:
        new_merged_pf = []
        for i in range(0, len(merged_pf[-1]), 2):
            new_merged_pf.append(merge_piecewise_linear(merged_pf[-1][i], merged_pf[-1][i+1]))
        if len(merged_pf[-1]) % 2 == 1:
            new_merged_pf.append(merged_pf[-1][-1])
        # Here we can shorten new_merged_points.
        merged_pf.append(new_merged_pf)
    
    return binary_search_max(merged_pf[-1])


if __name__ == "__main__":
    pt1 = XYPoint(-5, -2)
    pt2 = XYPoint(-4, 2)
    pt3 = XYPoint(0, 3)
    pt4 = XYPoint(3, 1)
    sg1 = PiecewiseSegment(None, pt1, 4)
    sg2 = PiecewiseSegment(pt1, pt3, 1)
    sg3 = PiecewiseSegment(pt3, None, -1)

    sg4 = PiecewiseSegment(None, pt2, 3)
    sg5 = PiecewiseSegment(pt2, pt4, -1/7)
    sg6 = PiecewiseSegment(pt4, None, -4)

    f1 = PiecewiseFunction(segments=[sg1, sg2, sg3])
    f2 = PiecewiseFunction(segments=[sg4, sg5, sg6])

    f1.plot()
    f2.plot()
    # plt.show()

    f3 = merge_piecewise_linear(f1, f2)
    f3.plot()
    plt.show()