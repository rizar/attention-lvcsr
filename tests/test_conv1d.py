from numpy.testing import assert_allclose

from lvsr.expressions import conv1d


def test_conv1d():
    a = [[1.0, 2, 3], [1, 0, 1]]
    b = [[2, 1], [1, 3.0]]
    c = conv1d(a, b).eval()
    assert_allclose(c, [[[5, 8], [5, 9]], [[1, 2], [3, 1]]])
    d = conv1d(a, b, border_mode='full').eval()
    assert_allclose(d, [[[2, 5, 8, 3], [1, 5, 9, 9]],
                        [[2, 1, 2, 1], [1, 3, 1, 3]]])
