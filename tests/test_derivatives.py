import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative
from scipy.spatial.transform import Rotation

import dragonET.command_line


@pytest.fixture
def simulated_data():
    np.random.seed(0)

    size = 512

    c = np.radians(np.arange(-10, 10 + 1, 2))
    dx = np.random.uniform(-50, 50, size=c.size)
    dy = np.random.uniform(-50, 50, size=c.size)
    a = np.radians(np.random.uniform(-10, 10, size=c.size))
    b = np.radians(np.random.uniform(-10, 10, size=c.size))

    handle = np.load("tests/template.npz")
    mask = handle["mask"][: c.size, :]
    mask = mask[:, np.count_nonzero(mask, axis=0) >= 4]

    Ra = Rotation.from_euler("z", a).as_matrix()
    Rb = Rotation.from_euler("x", b).as_matrix()
    Rc = Rotation.from_euler("y", c).as_matrix()
    R = Ra @ Rb @ Rc

    R = np.concatenate([R[:, 0, :], R[:, 1, :]], axis=0)
    S = np.random.uniform(-size // 4, size // 4, size=(3, mask.shape[1]))
    S -= np.mean(S, axis=1)[:, None]

    t = np.concatenate([dx + size // 2, dy + size // 2])

    W = R @ S + t[:, None]

    data = np.zeros((mask.shape[0], mask.shape[1], 2))
    data[:, :, 0] = W[: W.shape[0] // 2, :]
    data[:, :, 1] = W[W.shape[0] // 2 :, :]

    return data, mask, dx, dy, a, b, c


def derivative_test_function(d_func, indices, simulated_data):
    data, mask, dx, dy, a, b, c = simulated_data

    M = np.concatenate([mask, mask], axis=0)
    X = data[:, :, 0]
    Y = data[:, :, 1]
    W = np.concatenate([X, Y], axis=0)

    parameters = np.stack([dx, dy, a, b, c])
    active = np.zeros(parameters.shape, dtype=bool)
    active[indices, :] = 1

    def fun(x, parameters, active, W, M):
        parameters[active] = x
        return dragonET.command_line._refine.residuals(parameters, active, W, M)

    def jac(x, parameters, active, W, M):
        parameters[active] = x
        dx, dy, a, b, c = parameters

        # Get the translation
        t = np.concatenate([dx, dy], axis=0)

        # Subtract the translation
        W = W - t[:, None]

        # Call derivatives
        return d_func(dx, dy, a, b, c, W, M)

    x = parameters[active]

    J1 = jac(x, parameters, active, W, M)

    J0 = approx_derivative(fun, x, args=(parameters, active, W, M), method="3-point")

    assert np.max(np.abs(J0 - J1) < 1e-5)


def test_d_dt(simulated_data):
    derivative_test_function(dragonET.command_line._refine.d_dt, [0, 1], simulated_data)


def test_d_da(simulated_data):
    derivative_test_function(dragonET.command_line._refine.d_da, 2, simulated_data)


def test_d_db(simulated_data):
    derivative_test_function(dragonET.command_line._refine.d_db, 3, simulated_data)


def test_d_dc(simulated_data):
    derivative_test_function(dragonET.command_line._refine.d_dc, 4, simulated_data)


def test_jacobian(simulated_data):
    data, mask, dx, dy, a, b, c = simulated_data

    M = np.concatenate([mask, mask], axis=0)
    X = data[:, :, 0]
    Y = data[:, :, 1]
    W = np.concatenate([X, Y], axis=0)

    parameters = np.stack([dx, dy, a, b, c])
    active = np.ones(parameters.shape, dtype=bool)

    def fun(x, parameters, active, W, M):
        parameters[active] = x
        return dragonET.command_line._refine.residuals(parameters, active, W, M)

    def jac(x, parameters, active, W, M):
        parameters[active] = x
        return dragonET.command_line._refine.jacobian(parameters, active, W, M)

    x = parameters[active]

    J1 = jac(x, parameters, active, W, M)

    J0 = approx_derivative(fun, x, args=(parameters, active, W, M), method="3-point")

    assert np.max(np.abs(J0 - J1) < 1e-5)
