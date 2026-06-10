import numpy as np
import torch

from m3_learning.be.nn import SHO_fit_func_nn
from m3_learning.be.loop_fitter import loop_fitting_function_torch
from m3_learning.util.preprocessing import GlobalScaler
from m3_learning.nn.Fitter1D.Fitter1D import Multiscale1DFitter
from m3_learning.optimizers.AdaHessian import AdaHessian
from m3_learning.optimizers.TrustRegion import TRCG


def test_sho_fit_func_matches_numpy():
    A, w0, Q, phi = 1.0, 1.31e6, 100.0, 0.5
    params = torch.tensor([[A, w0, Q, phi]], dtype=torch.float64)
    wvec = np.linspace(1.2e6, 1.4e6, 165)
    out = SHO_fit_func_nn(params, wvec, device="cpu")
    expected = (A * np.exp(1j * phi) * w0**2) / (wvec**2 - 1j * wvec * w0 / Q - w0**2)
    np.testing.assert_allclose(out.numpy()[0], expected, rtol=1e-9)


def test_global_scaler_round_trip():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=3.0, scale=2.0, size=(50, 20))
    scaler = GlobalScaler()
    scaled = scaler.fit_transform(data.copy())
    assert abs(scaled.mean()) < 1e-9
    restored = scaler.inverse_transform(scaled.copy())
    np.testing.assert_allclose(restored, data, rtol=1e-9)


def test_loop_fitting_function_shape_cpu():
    V = np.concatenate([np.linspace(-10, 10, 48), np.linspace(10, -10, 48)])
    params = np.array([[0.1, 1.0, -2.0, 2.0, 0.01, 0.5, 1.5, 0.5, 1.5]])
    out = loop_fitting_function_torch(params, V, type="9 parameters", device="cpu")
    assert out.shape == (1, 96)
    assert torch.isfinite(out).all()


def test_multiscale1dfitter_forward_backward_cpu():
    wvec = np.linspace(1.2e6, 1.4e6, 165)
    model = Multiscale1DFitter(
        SHO_fit_func_nn, wvec, input_channels=2, num_params=4,
        scaler=None, post_processing=None, device="cpu",
    )
    model.train()
    x = torch.randn(8, 165, 2)
    out, params = model(x)
    assert params.shape == (8, 4)
    assert out.shape[0] == 8
    loss = (out.real ** 2 + out.imag ** 2).mean() if torch.is_complex(out) \
        else (out ** 2).mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_multiscale1dfitter_with_scaler_cpu_train_and_eval():
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(1)
    param_scaler = StandardScaler().fit(
        rng.normal(size=(100, 4)) * [1.0, 1e6, 50.0, 1.0] + [0, 1.3e6, 100.0, 0]
    )
    wvec = np.linspace(1.2e6, 1.4e6, 165)
    model = Multiscale1DFitter(
        SHO_fit_func_nn, wvec, input_channels=2, num_params=4,
        scaler=param_scaler, post_processing=None, device="cpu",
    )
    x = torch.randn(8, 165, 2)

    model.train()
    out, params = model(x)
    assert params.shape == (8, 4)

    model.eval()
    with torch.no_grad():
        out, embeddings, unscaled = model(x)
    assert embeddings.shape == (8, 4)
    assert unscaled.shape == (8, 4)


def _toy_problem():
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    X = torch.randn(64, 4)
    y = X @ torch.tensor([[1.0], [-2.0], [0.5], [3.0]]) + 0.1
    return model, X, y


def test_adahessian_step_decreases_loss():
    model, X, y = _toy_problem()
    opt = AdaHessian(model.parameters(), lr=0.1)
    loss0 = None
    for _ in range(20):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(X), y)
        if loss0 is None:
            loss0 = loss.item()
        loss.backward(create_graph=True)
        opt.step()
    assert loss.item() < loss0


def test_trcg_step_decreases_loss():
    model, X, y = _toy_problem()
    opt = TRCG(model, radius=1.0, device="cpu")

    def closure(part, total, device="cpu"):
        return torch.nn.functional.mse_loss(model(X), y)

    loss0 = closure(0, 1).item()
    for _ in range(5):
        opt.step(closure)
    assert closure(0, 1).item() < loss0
