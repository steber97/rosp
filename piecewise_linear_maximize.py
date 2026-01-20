from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from typing import TYPE_CHECKING, List, Tuple, Optional
from math import inf
import numpy as np
import matplotlib.pyplot as plt

from utils import EPS

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


def y(point: XYPoint, x: float, m: float) -> float:
    """
    Given a point and an angular coefficient, compute the y value corresponding
    to a certain x.
    Complexity O(1).
    """
    return m * (x - point.x) + point.y


def pt_intersect(p1: XYPoint, m1: float, p2: XYPoint, m2: float) -> XYPoint:
    """
    Given two points and two angular coefficients (they must be different),
    compute the point where the two lines intersect.
    Complexity O(1).
    """
    assert abs(m1 - m2) > EPS
    x = (-m2 * p2.x + m1 * p1.x - p1.y + p2.y) / (m1 - m2)
    y = m1 * (x - p1.x) + p1.y
    return XYPoint(x, y)

class PiecewiseSegment:
    def __init__(self, p1: Optional[XYPoint], p2: Optional[XYPoint], m: float):
        """
        If p1 or p2 are None, it means that the piecewise segment
        goes to or from +- infinity.
        At least one of them needs to be not None.
        If the angular coefficient does not correspond to the angular coefficient
        between points p1 and p2, an error is raised.
        The PiecewiseSegment cannot have x-length 0 (<EPS).
        
        :param p1: begin of segment.
        :param p2: end of segment.
        :param m: angular coefficient.
        """
        self.p1 = p1
        self.p2 = p2
        assert not (p1 is None and p2 is None)
        self.m = m
        assert abs(m) < EPS or abs(1/m) > EPS
        if self.p1 is not None and self.p2 is not None:
            assert p1.x <= p2.x and abs(p1.x - p2.x) > EPS
            assert abs((p2.x - p1.x) * m + p1.y - p2.y) < EPS


    def __repr__(self):
        return "{}, {}, m = {:.2f}".format(self.p1, self.p2, self.m)
    
    def _min_start_None(self, ps: PiecewiseSegment) -> List[PiecewiseSegment]:
        """
        Return the min of two segments aligned by x, s.t. they start from -infinity.
        I.e., their first point p1 is None.
        Time complexity O(1).
        
        :type ps: PiecewiseSegment
        :return: a list of one or two piecewise segments, starting with None,
                 that represents the min function between self and ps.
        """
        assert self.p1 is None and ps.p1 is None
        assert self.p2 is not None and ps.p2 is not None
        assert abs(self.p2.x - ps.p2.x) < EPS
        
        # If the angular coefficients are the same, then just keep the 
        # Segment with lower y value.
        if abs(self.m - ps.m) < EPS:
            if (self.p2.y <= ps.p2.y):
                return [self]
            else:
                return [ps]
        else:
            # If the angular coefficients are different, then the two segments (prolongued into a line) intersect.
            pt_inter = pt_intersect(self.p2, self.m, ps.p2, ps.m)
            
            if pt_inter.x >= self.p2.x or abs(pt_inter.x - self.p2.x) < EPS:
                # If the intersection point is after the end of the interval
                # Only keep the segment with the highest angular coefficient.
                if self.m > ps.m:
                    return [self]
                else:
                    return [ps]
            else:
                # The segment with highest angular coefficient appears first.
                # Second the other segment.
                if self.m > ps.m:
                    return [PiecewiseSegment(None, pt_inter, self.m),
                            PiecewiseSegment(pt_inter, ps.p2, ps.m)]
                else:
                    return [PiecewiseSegment(None, pt_inter, ps.m),
                            PiecewiseSegment(pt_inter, self.p2, self.m)]
    
    def _min_end_None(self, ps: PiecewiseSegment) -> List[PiecewiseSegment]:
        """
        Return the min of two segments aligned by x, s.t. they end with +infinity.
        I.e., their last point p2 is None.
        Time complexity O(1).
        
        :type ps: PiecewiseSegment
        :return: a list of one or two piecewise segments, ending with None,
                 that represents the min function between self and ps.
        """
        assert self.p2 is None and ps.p2 is None
        assert self.p1 is not None and ps.p1 is not None
        assert abs(self.p1.x - ps.p1.x) < EPS
        if abs(self.m - ps.m) < EPS:
            # If the angular coefficients are the same, then just keep the 
            # Segment with lower y value.
            if self.p1.y <= ps.p1.y:
                return [self]
            else:
                return [ps]
        else:
            # If the angular coefficients are different, then the two segments (prolongued into a line) intersect.
            pt_inter = pt_intersect(self.p1, self.m, ps.p1, ps.m)

            if pt_inter.x <= self.p1.x or abs(pt_inter.x - self.p1.x) < EPS:
                if self.m < ps.m:
                    return [self]
                else:
                    return [ps]
            else:
                if self.m < ps.m:
                    # The segment with highest angular coefficient appears first.
                    # Second the other segment.
                    return [PiecewiseSegment(ps.p1, pt_inter, ps.m),
                            PiecewiseSegment(pt_inter, None, self.m)]
                else:
                    return [PiecewiseSegment(self.p1, pt_inter, self.m),
                            PiecewiseSegment(pt_inter, None, ps.m)]
    
    def _min_no_None(self, ps: PiecewiseSegment) -> List[PiecewiseSegment]:
        """
        Given two piecewise segments (self and ps) such that they are aligned by x,
        and such that they have no endpoint set to None, return a list with 1 or 2 segments
        that represents the min of the two segments.
        Time complexity O(1).
        
        :param ps: Description
        :return: Description
        """
        assert self.p1 is not None and self.p2 is not None and ps.p1 is not None and ps.p2 is not None
        assert abs(self.p1.x - ps.p1.x) < EPS and abs(self.p2.x - ps.p2.x) < EPS
        assert abs(self.p1.x - self.p2.x) > EPS
        
        if ((self.p1.y < ps.p1.y) or abs(self.p1.y - ps.p1.y) < EPS) and (
            (self.p2.y < ps.p2.y) or abs(self.p2.y - ps.p2.y) < EPS):
            # If self is strictly below ps:
            return [self]
        elif ((self.p1.y > ps.p1.y) or abs(self.p1.y - ps.p1.y) < EPS) and (
            (self.p2.y > ps.p2.y) or abs(self.p2.y - ps.p2.y) < EPS):
            # If ps is strictly below self:
            return [ps]
        elif (self.p1.y < ps.p1.y) and (self.p2.y > ps.p2.y):
            # If self starts below, and then ps is below:
            pt_inter = pt_intersect(self.p1, self.m, ps.p1, ps.m)
            assert self.p1.x < pt_inter.x < self.p2.x
            segments = []
            if abs(self.p1.x - pt_inter.x) > EPS:
                # If the intersection doesn't happen on self.p1.x
                if abs(self.p2.x - pt_inter.x) > EPS:
                    # If it doesn't happen on p2
                    segments.append(PiecewiseSegment(self.p1, pt_inter, self.m))
                else:
                    segments.append(PiecewiseSegment(self.p1, self.p2, self.m))
            if abs(ps.p2.x - pt_inter.x) > EPS:
                if abs(self.p1.x - pt_inter.x) > EPS:
                    segments.append(PiecewiseSegment(pt_inter, ps.p2, ps.m)) 
                else:
                    segments.append(PiecewiseSegment(ps.p1, ps.p2, ps.m))
            return segments
        else:
            assert (self.p1.y > ps.p1.y) and (self.p2.y < ps.p2.y)
            pt_inter = pt_intersect(self.p1, self.m, ps.p1, ps.m)
            assert self.p1.x < pt_inter.x < self.p2.x
            segments = []
            if abs(pt_inter.x - ps.p1.x) > EPS:
                if abs(pt_inter.x - self.p2.x) < EPS:
                    segments.append(PiecewiseSegment(ps.p1, self.p2, ps.m))
                else:
                    segments.append(PiecewiseSegment(ps.p1, pt_inter, ps.m))
            if abs(pt_inter.x - ps.p2.x) > EPS:
                if abs(pt_inter.x - ps.p1.x) < EPS:
                    segments.append(PiecewiseSegment(ps.p1, self.p2, self.m))
                else:
                    segments.append(PiecewiseSegment(pt_inter, self.p2, self.m))
            return segments

    def min(self, ps: PiecewiseSegment) -> List[PiecewiseSegment]:
        """
        Given two piecewise segments, return the minimum of them. 
        This can be a piecewise function made of only one piecewise segment 
        (in case one of the two segments lies below the other)
        or a piecewise function of two segments (in case they cross at some point in the interval).
        Notice that the two segments should be defined over the same interval p1.x->p2.x. 
        Time complexity O(1).
        
        :param ps: other piecewise segment
        :return: A piecewise function (with at most 2 segments) encoding min(self, ps).
        """
        # Assert that the input makes sense.
        assert ((self.p1 is None and ps.p1 is None) or abs(self.p1.x - ps.p1.x) < EPS) \
            and ((self.p2 is None and ps.p2 is None) or abs(self.p2.x - ps.p2.x) < EPS)
        if self.p1 is not None and self.p2 is not None:
            assert self.p1.x < self.p2.x and abs(self.p1.x - self.p2.x) > EPS
        
        if self.p1 is None and ps.p1 is None:
            # Case when the two intervals start from -infty.
            return self._min_start_None(ps)

        elif self.p2 is None and ps.p2 is None:
            # Case when the two intervals end with -infty.
            return self._min_end_None(ps)   
        
        else:
            # General case.
            return self._min_no_None(ps)
    

class PiecewiseFunction:
    """
    Simple representation as consecutive piecewise segments.
    The function needs to start from -infinity (None) and end with +infinity (None).
    There need to be at least two piecewise segments.
    Consecutive Piecewise Segments need to be continuous and concave.
    """

    def __init__(self, segments: List[PiecewiseSegment]):
        """
        Time complexity O(n), where n is the number of segments.
        """
        self.segments: List[PiecewiseSegment] = segments
        assert len(segments) >= 2
        assert segments[0].p1 is None
        assert segments[-1].p2 is None 
        assert self.check_continuous()
        assert self.check_concave()
        for s in self.segments[1:-1]:
            assert s.p1.x < s.p2.x and abs(s.p1.x - s.p2.x) > EPS

    def check_continuous(self) -> bool:
        """
        Check that the piecewise function is continuous.
        Namely, the joints need to coincide.
        Time complexity O(n).
        """
        for i in range(len(self.segments) - 1):
            if self.segments[i].p2 != self.segments[i+1].p1:
                return False
        return True

    def check_convex(self) -> bool:
        """
        Check that the piecewise linear function is convex: namely,
        that the consecutive angular coefficients are non decreasing.
        Time complexity O(n).
        """
        if not self.check_continuous():
            return False
        for i in range(len(self.segments) - 1):
            if not(self.segments[i].m <= self.segments[i+1].m or abs(self.segments[i].m - self.segments[i+1].m) < EPS):
                return False
        return True

    def check_concave(self) -> bool:
        """
        Check that the piecewise linear function is concave: namely,
        that the consecutive angular coefficients are non increasing.
        Time complexity O(n).
        """
        if not self.check_continuous():
            return False
        for i in range(len(self.segments) - 1):
            if not(self.segments[i].m >= self.segments[i+1].m or abs(self.segments[i].m - self.segments[i+1].m) < EPS):
                return False
        return True

    def shorten(self) -> None:
        """
        If there are consecutive piecewise linear segments with the same angular coefficient,
        merge them into a unique segment.
        The operation happens inplace (self.segments is changed).
        Time complexity O(n).
        """
        new_segments: List[PiecewiseSegment] = []
        for i in range(len(self.segments)):
            if len(new_segments) == 0:
                new_segments.append(self.segments[i])
            # Special case: in case it is a straight line, then we still need 2 elements with None at the extremes.
            elif (abs(new_segments[-1].m - self.segments[i].m) < EPS) and not (i == len(self.segments) - 1 and len(new_segments) == 1):
                new_segments[-1].p2 = self.segments[i].p2
            else:
                new_segments.append(self.segments[i])
        self.segments = new_segments
    
    def plot(self) -> None:
        """
        Plot the piecewise linear function. 
        needs plt.show() in order to actually create the plot.
        """
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

def create_piecewise_linear(m: np.ndarray, v: np.ndarray, i: int) -> PiecewiseFunction:
    """
    Given the coefficients of the function f_i(x) = m[i] - x * v[i] - sum(|m[j] - x * v[j]|),
    return a Piecewise Linear function out of it.
    The complexity is O(n * log(n)).
    """

    # The idea is to sort the (m[j] - x * v[j]) by x when they switch sign.
    # Then, it is easy to compute the Piecewise linear segments in between the x's where the (...) switch sign.
    # We just keep track of the total x and const coefficient, and as soon as (...) switch sign,
    # we just update the total coefficient by that amount.
    tot_const = 0
    tot_x_coeff = 0
    a_s = [x for x in m]
    b_s = [x for x in v]

    # Make sure that the x coefficients are all positive.
    for j, (a, b) in enumerate(zip(a_s, b_s)):
        if b < 0:
            a_s[j] *= -1
            b_s[j] *= -1

    tot_const += m[i]
    tot_x_coeff -= v[i]
    for j in range(len(m)):
        if j != i:
            if abs(v[j]) < EPS:
                # If the x coefficient is zero, just subtract m[j]
                tot_const -= abs(m[j])
            else:
                tot_const -= a_s[j]
                tot_x_coeff -= -b_s[j]
    segments = []
    abx = [(a_s[j], b_s[j], a_s[j]/b_s[j]) for j, (a, b) in enumerate(zip(a_s, b_s)) if i != j and b_s[j] != 0]
    # Sort the intervals by increasing x (where the absolute value changes sign).
    abx = sorted(abx, key=lambda x: x[2])

    if len(abx) > 0:
        segments.append(PiecewiseSegment(None, XYPoint(abx[0][2], tot_const + abx[0][2] * tot_x_coeff), tot_x_coeff))
        for i, (a, b, x) in enumerate(abx[:-1]):
            # The absolute value changes sign, so we flip the coefficients.
            tot_const += 2*a
            tot_x_coeff -= 2*b
            # Prevent piecewise segment with empty x-interval.
            if abs(abx[i+1][2] - x) < EPS:
                segments[-1].p2.x = abx[i+1][2]
                segments[-1].p2.y = tot_const + abx[i+1][2] * tot_x_coeff
            else:
                segments.append(PiecewiseSegment(
                    XYPoint(x, tot_const + x * tot_x_coeff), 
                    XYPoint(abx[i+1][2], tot_const + abx[i+1][2] * tot_x_coeff),
                    tot_x_coeff
                ))
        # Add the last segment to +infinity.
        tot_const += 2*abx[-1][0]
        tot_x_coeff -= 2*abx[-1][1]
        segments.append(PiecewiseSegment(
            XYPoint(abx[-1][2], tot_const + abx[-1][2] * tot_x_coeff),
            None,
            tot_x_coeff
        ))
    else:
        # It is a straight line, but we need to add None at the beginning and at the end.
        segments.append(PiecewiseSegment(
            None,
            XYPoint(0, tot_const),
            tot_x_coeff
        ))
        segments.append(PiecewiseSegment(
            XYPoint(0, tot_const),
            None,
            tot_x_coeff
        ))

    return PiecewiseFunction(segments)


def min_piecewise_linear(l1: PiecewiseFunction, l2: PiecewiseFunction) -> PiecewiseFunction:
    """
    Given 2 piecewise linear functions,
    return a new piecewise linear function that is the min of the two.
    It is enough to split the two functions according to the same x's,
    and then understand which of the 2 functions is below in the aligned intervals.
    If they cross, add a new point.
    At the end, it would be good to remove useless intervals where the angular coefficient doesn't change.
    Time complexity O(|l1| + |l2|).

    :param l1:
    :param l2:
    :return:
    """

    def split_piecewise_function(l: PiecewiseFunction, xs: List[float]) -> List[PiecewiseSegment]:
        """
        Given a piecewise function, and a set of points xs, split the piecewise functions
        in segments that correspond to the xs.
        It should be the case that all the non-linear points in l are also contained in xs.
        Time complexity O(|l| + |xs|).
        
        :param l: 
        :param xs:
        :return:
        """
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

    # Compute the non-linearity points of the two functions. 
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

    for i in range(len(xs)-1):
        # Assert all intervals are non-empty.
        assert xs[i+1] > xs[i] and abs(xs[i+1] - xs[i]) > EPS

    seg1 = split_piecewise_function(l1, xs)
    seg2 = split_piecewise_function(l2, xs)

    segments = []
    # Compute the minimum of every pair of segments (aligned on the x-axis).
    for s1, s2 in zip(seg1, seg2):
        segments += s1.min(s2)

    f = PiecewiseFunction(segments)
    # If two consecutive segments have the same angular coefficient, merge them into a unique segment.
    f.shorten()
    return f

def binary_search_max(l: PiecewiseFunction) -> float:
    """
    Given a list of (points, angular coefficients) of a concave function,
    find the x that maximizes it. It is sufficient to look right as long as the angular coefficient is increasing,
    and to look left when the angular coefficient is decreasing.
    Time complexity O(log(|l|)).
    
    :param l: Description
    :type l: List[Tuple[Tuple[float, float], float]]
    :return: Description
    :rtype: float
    """
    assert l.check_concave()
    assert l.check_continuous()
    low = 0
    high = len(l.segments) - 1
    while low < high:
        mid = (high + low) // 2
        if abs(l.segments[mid].m) < EPS:
            # If the angular coefficient is zero, stop.
            return l.segments[mid].p1.x if l.segments[mid].p1 is not None else l.segments[mid].p2.x
        elif l.segments[mid].m < 0:
            high = mid
        else:
            low = mid + 1
    assert low == high
    if abs(l.segments[low].m) < EPS:
        return l.segments[low].p1.x if l.segments[low].p1 is not None else l.segments[low].p2.x
    elif l.segments[low].m > 0:
        return l.segments[low].p2.x if l.segments[low].p2 is not None else inf
    else:
        assert l.segments[low].m < 0
        return l.segments[low].p1.x if l.segments[low].p1 is not None else -inf

def worker(args):
    i, m_row, v_row = args
    return create_piecewise_linear(m_row, v_row, i)

def maximize_x(M: np.ndarray, V: np.ndarray) -> float:
    """
    Given the coefficients M, V, find the x that maximizes
    the diagonally dominance condition of M - x*V.
    We do it using a "mergesort" strategy:
    The diagonally dominance condition on every row of the matrix M - xV 
    is a concave piecewise linear function: dd(i) = M[i] - x*v[i] - sum_{j!=i} |m[j] - x v[j]|.
    The diagonally dominance condition of M - x V is the min_i dd(i).
    We are looking for max_x min_i dd(i).
    We do it by computing the min_i dd(i) function for pairs of rows recursively.
    When we have the global min function, we can use binary search to look for the maximizer x.
    This takes only O(n^2 log(n)).
    
    :param M: 
    :param V:
    """

    tasks = [(i, M[i, :], V[i, :]) for i in range(len(M))]
    merged_pf = [[]] # merged piecewise functions
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(worker, tasks))
    merged_pf[0].extend(results)
    while len(merged_pf[-1]) > 1:
        assert sum([len(merged_pf[-1][i].segments) for i in range(len(merged_pf[-1]))]) <= 2*len(M)**2
        new_merged_pf = []
        # Merge piecewise linear function in pairs.
        for i in range(0, len(merged_pf[-1])-1, 2):
            new_merged_pf.append(min_piecewise_linear(merged_pf[-1][i], merged_pf[-1][i+1]))
        if len(merged_pf[-1]) % 2 == 1:
            new_merged_pf.append(merged_pf[-1][-1])
        merged_pf.append(new_merged_pf)
    assert len(merged_pf) < 2 * np.log2(len(M))
    for pf_list in merged_pf:
        assert np.sum([len(pf.segments) for pf in pf_list]) <= (len(M)**2)
    # Return the maximizer x. If it is negative, return 0 instead.
    return max(0, binary_search_max(merged_pf[-1][0]))
