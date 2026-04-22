#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <optional>
#include <stdexcept>

// c++ -O3 -Wall -shared -std=c++17 -fPIC $(python3 -m pybind11 --includes) -undefined dynamic_lookup module.cpp -o optimization_module$(python3-config --extension-suffix)

namespace py = pybind11;

const double EPS = 1e-9;
const double INF = std::numeric_limits<double>::infinity();

struct XYPoint {
    double x, y;
    bool operator==(const XYPoint& other) const {
        return std::abs(x - other.x) < EPS && std::abs(y - other.y) < EPS;
    }
    bool operator!=(const XYPoint& other) const { return !(*this == other); }
};

double get_y(XYPoint p, double x, double m) {
    // Given a line as point p and angular coefficient m,
    // compute the y coordinate for coordinate x.
    return m * (x - p.x) + p.y;
}

XYPoint pt_intersect(XYPoint p1, double m1, XYPoint p2, double m2) {
    if (std::abs(m1 - m2) < EPS) throw std::runtime_error("Parallel lines do not intersect");
    double x = (-m2 * p2.x + m1 * p1.x - p1.y + p2.y) / (m1 - m2);
    double y = m1 * (x - p1.x) + p1.y;
    return {x, y};
}

struct PiecewiseSegment {
    std::optional<XYPoint> p1;
    std::optional<XYPoint> p2;
    double m;

    PiecewiseSegment(std::optional<XYPoint> _p1, std::optional<XYPoint> _p2, double _m) 
        : p1(_p1), p2(_p2), m(_m) {}

    // Method implementations for min calculation
    std::vector<PiecewiseSegment> _min_start_None(const PiecewiseSegment& ps) const {
        XYPoint self_p2 = *p2, ps_p2 = *ps.p2;
        if (std::abs(m - ps.m) < EPS) {
            return (self_p2.y <= ps_p2.y) ? std::vector<PiecewiseSegment>{*this} : std::vector<PiecewiseSegment>{ps};
        }
        XYPoint inter = pt_intersect(self_p2, m, ps_p2, ps.m);
        if (inter.x >= self_p2.x - EPS) {
            return (m > ps.m) ? std::vector<PiecewiseSegment>{*this} : std::vector<PiecewiseSegment>{ps};
        }
        if (m > ps.m) return {{std::nullopt, inter, m}, {inter, ps_p2, ps.m}};
        else return {{std::nullopt, inter, ps.m}, {inter, self_p2, m}};
    }

    std::vector<PiecewiseSegment> _min_end_None(const PiecewiseSegment& ps) const {
        XYPoint self_p1 = *p1, ps_p1 = *ps.p1;
        if (std::abs(m - ps.m) < EPS) {
            return (self_p1.y <= ps_p1.y) ? std::vector<PiecewiseSegment>{*this} : std::vector<PiecewiseSegment>{ps};
        }
        XYPoint inter = pt_intersect(self_p1, m, ps_p1, ps.m);
        if (inter.x <= self_p1.x + EPS) {
            return (m < ps.m) ? std::vector<PiecewiseSegment>{*this} : std::vector<PiecewiseSegment>{ps};
        }
        if (m < ps.m) return {{ps_p1, inter, ps.m}, {inter, std::nullopt, m}};
        else return {{self_p1, inter, m}, {inter, std::nullopt, ps.m}};
    }

    std::vector<PiecewiseSegment> _min_no_None(const PiecewiseSegment& ps) const {
        XYPoint s1 = *p1, s2 = *p2, o1 = *ps.p1, o2 = *ps.p2;
        // static_assert(std::abs(s1.x - o1.x) < EPS);
        bool self_starts_below = s1.y <= o1.y + EPS;
        bool self_ends_below = s2.y <= o2.y + EPS;

        if (self_starts_below && self_ends_below) return {*this};
        if (!self_starts_below && !self_ends_below) return {ps};

        XYPoint inter = pt_intersect(s1, m, o1, ps.m);
        std::vector<PiecewiseSegment> res;
        if (std::abs(s1.x - inter.x) > EPS && std::abs(s2.x - inter.x) > EPS) {
            if (self_starts_below) {
                return {
                    PiecewiseSegment(s1, inter, m), PiecewiseSegment(inter, o2, ps.m)
                };
            }
            else {
                return {
                    PiecewiseSegment(o1, inter, ps.m), PiecewiseSegment(inter, s2, m)
                };
            }
        }
        else {
            if (std::abs(s1.x - inter.x) <= EPS) {
                if (m <= ps.m) {
                    return {*this};
                }
                else {
                    return {ps};
                }
            }
            else {
                if (m <= ps.m) {
                    return {ps};
                }
                else {
                    return {*this};
                }
            }
        }
    }

    std::vector<PiecewiseSegment> min_seg(const PiecewiseSegment& ps) const {
        if (!p1 && !ps.p1) return _min_start_None(ps);
        if (!p2 && !ps.p2) return _min_end_None(ps);
        return _min_no_None(ps);
    }
};

class PiecewiseFunction {
public:
    std::vector<PiecewiseSegment> segments;

    PiecewiseFunction(std::vector<PiecewiseSegment> segs) : segments(segs) {}

    void shorten() {
        if (segments.empty()) return;
        std::vector<PiecewiseSegment> news;
        for (auto& s : segments) {
            if (news.empty()) news.push_back(s);
            else if (std::abs(news.back().m - s.m) < EPS && !(news.size() == 1 && !s.p2)) {
                news.back().p2 = s.p2;
            } else {
                news.push_back(s);
            }
        }
        segments = news;
    }

    bool check_concave() const {
        for (size_t i = 0; i < segments.size() - 1; ++i) {
            if (segments[i].m < segments[i+1].m - EPS) return false;
        }
        return true;
    }
};

// --- Logic Functions ---

PiecewiseFunction create_piecewise_linear(py::array_t<double> m_arr, py::array_t<double> v_arr, int row_idx) {
    auto m = m_arr.unchecked<1>();
    auto v = v_arr.unchecked<1>();
    ssize_t n = m.shape(0);

    double tot_const = m(row_idx);
    double tot_x_coeff = -v(row_idx);

    struct ABX { double a, b, x; };
    std::vector<ABX> abx;

    for (int j = 0; j < n; ++j) {
        if (j == row_idx) continue;
        double a = m(j), b = v(j);
        if (b < -EPS) { a *= -1; b *= -1; }
        
        if (std::abs(b) < EPS) {
            tot_const -= std::abs(a);
        } else {
            tot_const -= a;
            tot_x_coeff += b;
            abx.push_back({a, b, a/b});
        }
    }

    std::sort(abx.begin(), abx.end(), [](const ABX& l, const ABX& r) { return l.x < r.x; });

    std::vector<PiecewiseSegment> segments;
    if (!abx.empty()) {
        segments.emplace_back(std::nullopt, XYPoint{abx[0].x, tot_const + abx[0].x * tot_x_coeff}, tot_x_coeff);
        for (size_t k = 0; k < abx.size() - 1; ++k) {
            tot_const += 2 * abx[k].a;
            tot_x_coeff -= 2 * abx[k].b;
            if (std::abs(abx[k+1].x - abx[k].x) > EPS) {
                XYPoint p1{abx[k].x, tot_const + abx[k].x * tot_x_coeff};
                XYPoint p2{abx[k+1].x, tot_const + abx[k+1].x * tot_x_coeff};
                segments.emplace_back(p1, p2, tot_x_coeff);
            }
        }
        tot_const += 2 * abx.back().a;
        tot_x_coeff -= 2 * abx.back().b;
        segments.emplace_back(XYPoint{abx.back().x, tot_const + abx.back().x * tot_x_coeff}, std::nullopt, tot_x_coeff);
    } else {
        segments.emplace_back(std::nullopt, XYPoint{0, tot_const}, tot_x_coeff);
        segments.emplace_back(XYPoint{0, tot_const}, std::nullopt, tot_x_coeff);
    }
    return PiecewiseFunction(segments);
}

PiecewiseFunction min_piecewise_linear(const PiecewiseFunction& l1, const PiecewiseFunction& l2) {
    std::vector<double> xs;
    size_t i = 1;
    size_t j = 1;
    
    while(i < l1.segments.size() && j < l2.segments.size()) {
        if (std::abs(l1.segments[i].p1->x - l2.segments[j].p1->x) < EPS) {
            xs.push_back(l1.segments[i].p1->x);
            i++;
            j++;
        }
        else if (l1.segments[i].p1->x < l2.segments[j].p1->x) {
            xs.push_back(l1.segments[i].p1->x);
            i++;
        }
        else {
            xs.push_back(l2.segments[j].p1->x);
            j++;
        }
    }
    
    while (i < l1.segments.size()) {
        xs.push_back(l1.segments[i].p1->x);
        i++;
    }
    while(j < l2.segments.size()) {
        xs.push_back(l2.segments[j].p1->x);
        j++;
    }

    auto split = [](const PiecewiseFunction& f, const std::vector<double>& points) {
        std::vector<PiecewiseSegment> res;
        size_t cur_seg = 0;
        
        // First segment (-inf to points[0])
        double m0 = f.segments[0].m;
        XYPoint p_end = *f.segments[0].p2;
        res.emplace_back(std::nullopt, XYPoint{points[0], get_y(p_end, points[0], m0)}, m0);
        
        if (std::abs(points[0] - p_end.x) < EPS) cur_seg++;

        for (size_t k = 0; k < points.size() - 1; ++k) {
            double mk = f.segments[cur_seg].m;
            XYPoint ref = f.segments[cur_seg].p1 ? *f.segments[cur_seg].p1 : *f.segments[cur_seg].p2;
            res.emplace_back(XYPoint{points[k], get_y(ref, points[k], mk)}, 
                             XYPoint{points[k+1], get_y(ref, points[k+1], mk)}, mk);
            if (f.segments[cur_seg].p2 && std::abs(points[k+1] - f.segments[cur_seg].p2->x) < EPS) cur_seg++;
        }
        
        // Last segment
        double m_last = f.segments.back().m;
        XYPoint p_start = *f.segments.back().p1;
        res.emplace_back(XYPoint{points.back(), get_y(p_start, points.back(), m_last)}, std::nullopt, m_last);
        return res;
    };

    auto segs1 = split(l1, xs);
    auto segs2 = split(l2, xs);
    std::vector<PiecewiseSegment> final_segs;
    for (size_t i = 0; i < segs1.size(); ++i) {
        auto m = segs1[i].min_seg(segs2[i]);
        final_segs.insert(final_segs.end(), m.begin(), m.end());
    }
    PiecewiseFunction f(final_segs);
    f.shorten();
    return f;
}

double binary_search_max(const PiecewiseFunction& l) {
    int low = 0, high = l.segments.size() - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        if (std::abs(l.segments[mid].m) < EPS) return l.segments[mid].p1 ? l.segments[mid].p1->x : l.segments[mid].p2->x;
        if (l.segments[mid].m < 0) high = mid;
        else low = mid + 1;
    }
    if (std::abs(l.segments[low].m) < EPS) return l.segments[low].p1 ? l.segments[low].p1->x : l.segments[low].p2->x;
    if (l.segments[low].m > 0) return l.segments[low].p2 ? l.segments[low].p2->x : INF;
    return l.segments[low].p1 ? l.segments[low].p1->x : -INF;
}

double maximize_x_cpp(py::array_t<double> M, py::array_t<double> V) {
    auto m_ref = M.unchecked<2>();
    int n = m_ref.shape(0);
    
    std::vector<PiecewiseFunction> pfs;
    for (int i = 0; i < n; ++i) {
        // Extract row as array_t
        py::array_t<double> m_row(n), v_row(n);
        auto mr = m_row.mutable_unchecked<1>();
        auto vr = v_row.mutable_unchecked<1>();
        for(int j=0; j<n; j++) { mr(j) = m_ref(i,j); vr(j) = V.at(i,j); }
        
        pfs.push_back(create_piecewise_linear(m_row, v_row, i));
    }

    while (pfs.size() > 1) {
        std::vector<PiecewiseFunction> next_pfs;
        for (size_t i = 0; i < pfs.size() - 1; i += 2) {
            next_pfs.push_back(min_piecewise_linear(pfs[i], pfs[i+1]));
        }
        if (pfs.size() % 2 == 1) next_pfs.push_back(pfs.back());
        pfs = std::move(next_pfs);
    }

    return std::max(0.0, binary_search_max(pfs[0]));
}

PYBIND11_MODULE(optimization_module, m) {
    m.def("maximize_x_cpp", &maximize_x_cpp, "Finds x that maximizes diagonally dominance");
}