import numpy as np
from n2v.utils import n2v_utils
from n2v.utils.n2v_utils import tta_forward, tta_backward


def test_get_subpatch():
    patch = np.arange(100)
    patch.shape = (10, 10)

    subpatch_target = np.array([[11, 12, 13, 14, 15],
                                [21, 22, 23, 24, 25],
                                [31, 32, 33, 34, 35],
                                [41, 42, 43, 44, 45],
                                [51, 52, 53, 54, 55]])

    subpatch_test = n2v_utils.get_subpatch(patch, (3, 3), 2)

    assert np.sum(subpatch_target - subpatch_test) == 0

    subpatch_test = n2v_utils.get_subpatch(patch, (3, 3), 1)

    assert np.sum(subpatch_target[1:-1, 1:-1] - subpatch_test) == 0

    patch = np.arange(1000)
    patch.shape = (10, 10, 10)

    subpatch_target = np.array([[[31, 32, 33],
                                 [41, 42, 43],
                                 [51, 52, 53]],
                                [[131, 132, 133],
                                 [141, 142, 143],
                                 [151, 152, 153]],
                                [[231, 232, 233],
                                 [241, 242, 243],
                                 [251, 252, 253]]])

    subpatch_test = n2v_utils.get_subpatch(patch, (1, 4, 2), 1)

    assert np.sum(subpatch_target - subpatch_test) == 0


def test_random_neighbor():
    coord = np.array([51, 52, 32])
    shape = [128, 128, 128]

    for i in range(1000):
        coords = n2v_utils.random_neighbor(shape, coord)
        assert np.all(coords != coord)

    shape = [55, 53, 32]

    for i in range(1000):
        coords = n2v_utils.random_neighbor(shape, coord)
        assert np.all(coords != coord)


def test_pm_normal_neighbor_withoutCP():
    patch = np.arange(100)
    patch.shape = (10, 10)

    coords = (np.array([2, 4]), np.array([1, 3]))

    sampler = n2v_utils.pm_normal_withoutCP(1)

    for i in range(100):
        val = sampler(patch, coords, len(patch.shape))
        for v in val:
            assert 0 <= v and v < 100

    patch = np.arange(1000)
    patch.shape = (10, 10, 10, 1)

    coords = (np.array([2, 4, 6]), np.array([1, 3, 5]), np.array([3, 5, 1]))

    for i in range(100):
        val = sampler(patch, coords, len(patch.shape))
        for v in val:
            assert 0 <= v and v < 1000


def test_pm_uniform_withCP():
    patch = np.arange(100)
    patch.shape = (10, 10)

    coords = (np.array([2, 4]), np.array([1, 3]))

    sampler = n2v_utils.pm_uniform_withCP(3)

    for i in range(100):
        val = sampler(patch, coords, len(patch.shape))
        for v in val:
            assert 0 <= v and v < 100

    patch = np.arange(1000)
    patch.shape = (10, 10, 10)

    coords = (np.array([2, 4, 6]), np.array([1, 3, 5]), np.array([3, 5, 1]))

    for i in range(10):
        val = sampler(patch, coords, len(patch.shape))
        for v in val:
            assert 0 <= v and v < 1000


def test_pm_normal_additive():
    patch = np.arange(100)
    patch.shape = (10, 10)

    coords = (np.array([2, 4]), np.array([1, 3]))

    sampler = n2v_utils.pm_normal_additive(0)

    val = sampler(patch, coords, len(patch.shape))
    for v, y, x in zip(val, *coords):
        assert v == patch[y, x]

    patch = np.arange(1000)
    patch.shape = (10, 10, 10)

    coords = (np.array([2, 4, 6]), np.array([1, 3, 5]), np.array([3, 5, 1]))

    val = sampler(patch, coords, len(patch.shape))
    for v, z, y, x in zip(val, *coords):
        assert v == patch[z, y, x]


def test_pm_normal_fitted():
    patch = np.arange(100)
    patch.shape = (10, 10)

    coords = (np.array([2, 4]), np.array([1, 3]))

    sampler = n2v_utils.pm_normal_fitted(3)

    val = sampler(patch, coords, len(patch.shape))
    for v in val:
        assert isinstance(v, float)

    patch = np.arange(1000)
    patch.shape = (10, 10, 10)

    coords = (np.array([2, 4, 6]), np.array([1, 3, 5]), np.array([3, 5, 1]))

    val = sampler(patch, coords, len(patch.shape))
    for v in val:
        assert isinstance(v, float)


def test_pm_identity():
    patch = np.arange(100)
    patch.shape = (10, 10)

    coords = (np.array([2, 4]), np.array([1, 3]))
    sampler = n2v_utils.pm_identity(1)

    val = sampler(patch, coords, len(patch.shape))
    for v, y, x in zip(val, *coords):
        assert v == patch[y, x]

    patch = np.arange(1000)
    patch.shape = (10, 10, 10, 1)

    coords = (np.array([2, 4, 6]), np.array([1, 3, 5]), np.array([3, 5, 1]))

    val = sampler(patch, coords, len(patch.shape))
    for v, z, y, x in zip(val, *coords):
        assert v == patch[z, y, x]


def test_tta():
    img, _ = np.meshgrid(range(200), range(100))
    img[:50, :50] = 50
    aug = tta_forward(img[..., np.newaxis])
    avg = tta_backward(aug)
    assert np.sum(avg[..., 0] - img) == 0
