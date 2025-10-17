"""Dynamics definitions and utilities for continuous-time neuromorphic systems (C-IDEN).

This module implements the core continuous-time dynamics with:
- Stable linear generator parameterization A = -(alpha I + L L^T) enforcing stability when enabled
- Nonlinear ODE field f(h, t) = A h + rho(W_h h + b [+ W_x x(t)])
- Differentiable jump update U(x_i, h_minus) via a small MLP returning delta h
- Intensity head lambda*(t) = g(W_o h + b_o) with positivity enforcement
- A pure-PyTorch fixed-step integrator (RK4 by default; Euler fallback if dt is tiny)

Training–inference decoupling: This module computes continuous dynamics and jump updates.
It does NOT perform spike sampling nor thinning; those are handled at higher levels.

Minimal example (not executed):

    import torch
    from sr_ciden.dynamics import LinearNonlinearODE, JumpMLP, IntensityHead, integrate

    D, X = 8, 3
    ode = LinearNonlinearODE(
        dim=D,
        x_dim=X,
        nonlinearity="softplus",
        use_stable_A=True,
        alpha_init=0.05,
    )

    # Optionally provide external input function x(t)
    def x_of_t(t: torch.Tensor) -> torch.Tensor:
        # Return [batch, X] features aligned with the current batch
        # For illustration, zeros:
        batch = 4
        return torch.zeros(batch, X, device=t.device, dtype=t.dtype)

    # Optional jump update module
    jump = JumpMLP(state_dim=D, input_dim=X, hidden_dim=2 * D, nonlinearity="tanh")

    # Intensity head mapping state -> intensity
    head = IntensityHead(state_dim=D, activation="softplus")

    # ODE callable captures parameters via the Module
    def ode_func(h: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # If the ODE needs x(t), provide x_of_t to the module once:
        ode.set_input_provider(x_of_t)
        return ode(h, t)

    h0 = torch.zeros(4, D)
    event_times = torch.tensor([0.5, 1.2, 2.0])  # strictly increasing

    traj = integrate(
        ode_func=ode_func,
        h0=h0,
        event_times=event_times,
        jump_update=jump,  # callable(x_i, h_minus) -> delta h
        x_of_t=x_of_t,     # supplies x_i at each event time
        dt=1e-3,
        T_end=2.0,
        return_every_step=False,
    )

Note on disabling stability reparameterization:
If `use_stable_A=False`, this module exposes a free A parameter with no constraints.
A small spectral penalty (not implemented here) is recommended externally to discourage
exploding dynamics.

No top-level code runs on import.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Protocol, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Tensor = torch.Tensor
Scalar = Union[float, int, Tensor]


def _get_nonlinearity(name: str) -> Callable[[Tensor], Tensor]:
    """Return a pointwise nonlinearity."""
    key = name.lower()
    if key == "softplus":
        return F.softplus
    if key == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported nonlinearity '{name}'. Choose from ['softplus','tanh'].")


class _ODEFuncLike(Protocol):
    """Protocol for ODE callables."""

    def __call__(self, h: Tensor, t: Tensor) -> Tensor:  # pragma: no cover - protocol
        ...


class _JumpLike(Protocol):
    """Protocol for jump-update callables."""

    def __call__(self, x_i: Optional[Tensor], h_minus: Tensor) -> Tensor:  # pragma: no cover - protocol
        ...


@dataclass
class Trajectory:
    """Light-weight container for integrated trajectories.

    Attributes
    ----------
    times:
        1D tensor of shape [T_steps_total]. Always includes t=0 and T_end. If event
        times were provided, those will also be included.
    states:
        Tensor of shape [T_steps_total, batch, D] aligned with `times`.
    marks:
        Optional tensor of marks aligned with event times if passed to integrate in
        higher-level APIs. This integrator does not populate marks and will return None.
    """

    times: Tensor
    states: Tensor
    marks: Optional[Tensor] = None


class StableMatrixParam(nn.Module):
    """Stable matrix parameterization for linear generator A.

    When `stable=True`, constructs:
        A = -( alpha * I + L L^T ), with alpha = softplus(alpha_raw) + eps > 0
    which guarantees that A is negative definite (or at least negative semidefinite,
    depending on L). This typically yields stable (contractive) dynamics.

    When `stable=False`, exposes a free, unconstrained parameter A of shape [D, D].
    A small spectral penalty is recommended externally to discourage exploding
    dynamics, but no penalty is implemented here.

    Parameters
    ----------
    dim:
        State dimensionality D.
    stable:
        If True, use the stable reparameterization. If False, expose unconstrained A.
    alpha_init:
        Initial value for alpha (before softplus). The effective alpha at init will be
        softplus(alpha_init). This will be further offset by a tiny epsilon to ensure
        strict positivity.

    Notes
    -----
    - No randomness is required by this class; parameters are initialized to zeros
      except `alpha_raw` which is set to `alpha_init`.
    """

    def __init__(self, dim: int, stable: bool = True, alpha_init: float = 0.05) -> None:
        super().__init__()
        self.dim = int(dim)
        self.stable = bool(stable)

        if self.stable:
            # alpha strictly positive via softplus
            self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))
            # Raw factor; effective factor is tril(L_raw)
            self.L_raw = nn.Parameter(torch.zeros(self.dim, self.dim))
        else:
            self.A_free = nn.Parameter(torch.zeros(self.dim, self.dim))
            self.register_parameter("alpha_raw", None)
            self.register_parameter("L_raw", None)

        # Small epsilon to ensure strict positiveness for alpha
        self._alpha_eps: float = 1e-8

    def forward(self) -> Tensor:
        """Return the current A matrix of shape [D, D]."""
        if not self.stable:
            return self.A_free

        alpha = F.softplus(self.alpha_raw) + self._alpha_eps
        L = torch.tril(self.L_raw)
        A = -(alpha * torch.eye(self.dim, device=L.device, dtype=L.dtype) + L @ L.T)
        return A

    @property
    def alpha(self) -> Optional[Tensor]:
        """Current positive alpha or None if `stable=False`."""
        if not self.stable:
            return None
        return F.softplus(self.alpha_raw) + self._alpha_eps


class LinearNonlinearODE(nn.Module):
    """Nonlinear ODE field f(h, t) = A h + rho(W_h h + b [+ W_x x(t)]).

    Parameters
    ----------
    dim:
        State dimensionality D.
    x_dim:
        Optional external input dimensionality X. If > 0, enables W_x.
    nonlinearity:
        Choice of rho. One of {'softplus', 'tanh'}. Default 'softplus'.
    use_stable_A:
        If True, parameterize A = -(alpha I + L L^T). Else expose a free A.
    alpha_init:
        Initial value for alpha (before softplus) when `use_stable_A=True`.

    Notes
    -----
    - Call `set_input_provider(x_of_t)` once if you want the ODE to use an external
      input function inside its forward pass.
    - For training/inference decoupling, this module is agnostic to any sampling or
      readout mechanism.
    """

    def __init__(
        self,
        dim: int,
        x_dim: int = 0,
        nonlinearity: str = "softplus",
        use_stable_A: bool = True,
        alpha_init: float = 0.05,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.x_dim = int(x_dim)
        self.rho = _get_nonlinearity(nonlinearity)
        self.A_param = StableMatrixParam(dim=self.dim, stable=use_stable_A, alpha_init=alpha_init)

        # Affine block W_h h + b
        self.W_h = nn.Parameter(torch.zeros(self.dim, self.dim))
        self.b_h = nn.Parameter(torch.zeros(self.dim))

        # Optional external drive W_x x(t)
        if self.x_dim > 0:
            self.W_x = nn.Parameter(torch.zeros(self.dim, self.x_dim))
        else:
            self.register_parameter("W_x", None)

        self._x_provider: Optional[Callable[[Tensor], Tensor]] = None

    def set_input_provider(self, fn: Optional[Callable[[Tensor], Tensor]]) -> None:
        """Set an optional input provider x_of_t(t) -> [batch, x_dim]."""
        self._x_provider = fn

    def forward(self, h: Tensor, t: Tensor) -> Tensor:
        """Compute f(h, t).

        Parameters
        ----------
        h:
            Tensor of shape [batch, D].
        t:
            Scalar time as a 0-d tensor (device- and dtype-aligned with `h`).

        Returns
        -------
        dh_dt:
            Tensor of shape [batch, D].
        """
        if h.ndim != 2 or h.shape[1] != self.dim:
            raise ValueError(f"Expected h of shape [batch, {self.dim}], got {tuple(h.shape)}")
        A = self.A_param()
        Ah = h @ A.T  # [B, D]

        z = h @ self.W_h.T + self.b_h  # [B, D]
        if self.W_x is not None and self._x_provider is not None:
            x_t = self._x_provider(t)  # expected [B, x_dim]
            if x_t is None:
                # Treat as zeros if provider returns None
                x_term = 0.0
            else:
                if x_t.ndim != 2 or x_t.shape[1] != self.x_dim:
                    raise ValueError(
                        f"x_of_t(t) must return [batch, {self.x_dim}], got {tuple(x_t.shape)}"
                    )
                x_term = x_t @ self.W_x.T  # [B, D]
            z = z + x_term

        return Ah + self.rho(z)


class JumpMLP(nn.Module):
    """Two-layer MLP jump update U(x_i, h_minus) -> delta h.

    h_plus = h_minus + U(x_i, h_minus)

    Parameters
    ----------
    state_dim:
        State dimension D (output dimension).
    input_dim:
        External mark/features dimension X for jump input x_i. If 0, only h_minus is used.
    hidden_dim:
        Hidden layer width. Default: max(32, 2 * D).
    nonlinearity:
        Activation between the two linear layers. One of {'softplus', 'tanh'}.
    """

    def __init__(
        self,
        state_dim: int,
        input_dim: int = 0,
        hidden_dim: Optional[int] = None,
        nonlinearity: str = "tanh",
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.input_dim = int(input_dim)
        hidden = hidden_dim if hidden_dim is not None else max(32, 2 * self.state_dim)
        self.rho = _get_nonlinearity(nonlinearity)

        in_dim = self.state_dim + (self.input_dim if self.input_dim > 0 else 0)
        self.lin1 = nn.Linear(in_dim, hidden)
        self.lin2 = nn.Linear(hidden, self.state_dim)

    def forward(self, x_i: Optional[Tensor], h_minus: Tensor) -> Tensor:
        """Compute delta h for a jump at an event time.

        Parameters
        ----------
        x_i:
            Event mark/features of shape [batch, X] (or None).
        h_minus:
            Pre-jump hidden state of shape [batch, D].

        Returns
        -------
        delta_h:
            Tensor of shape [batch, D].
        """
        if h_minus.ndim != 2 or h_minus.shape[1] != self.state_dim:
            raise ValueError(f"h_minus must be [batch, {self.state_dim}], got {tuple(h_minus.shape)}")

        if self.input_dim > 0:
            if x_i is None:
                x_i = torch.zeros(h_minus.shape[0], self.input_dim, device=h_minus.device, dtype=h_minus.dtype)
            if x_i.ndim != 2 or x_i.shape[1] != self.input_dim:
                raise ValueError(f"x_i must be [batch, {self.input_dim}], got {tuple(x_i.shape)}")
            z = torch.cat([h_minus, x_i], dim=-1)
        else:
            z = h_minus

        return self.lin2(self.rho(self.lin1(z)))


class IntensityHead(nn.Module):
    """Map hidden state to nonnegative intensity lambda*(t).

    Parameters
    ----------
    state_dim:
        State dimension D.
    activation:
        Output transfer g. One of {'softplus', 'exp'}. Default 'softplus'.
    epsilon:
        Small positive value added to avoid exact zeros.
    exp_clip_max:
        Optional clamp maximum applied to pre-exponential logits when activation='exp'
        to mitigate overflow.

    Returns
    -------
    Forward returns a tensor of shape [batch, 1].
    """

    def __init__(
        self,
        state_dim: int,
        activation: str = "softplus",
        epsilon: float = 1e-8,
        exp_clip_max: Optional[float] = 20.0,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.W_o = nn.Linear(self.state_dim, 1, bias=True)
        self.activation = activation.lower()
        self.epsilon = float(epsilon)
        self.exp_clip_max = exp_clip_max

        if self.activation not in {"softplus", "exp"}:
            raise ValueError("activation must be one of {'softplus','exp'}")

    def forward(self, h: Tensor) -> Tensor:
        """Compute lambda*(t) given h(t)."""
        z = self.W_o(h)  # [B, 1]
        if self.activation == "softplus":
            y = F.softplus(z)
        else:  # 'exp'
            if self.exp_clip_max is not None:
                z = torch.clamp(z, max=float(self.exp_clip_max))
            y = torch.exp(z)
        return y + self.epsilon


def integrate(
    ode_func: _ODEFuncLike,
    h0: Tensor,
    event_times: Tensor,
    jump_update: Optional[_JumpLike] = None,
    x_of_t: Optional[Callable[[Tensor], Tensor]] = None,
    dt: float = 1e-3,
    T_end: Optional[float] = None,
    return_every_step: bool = False,
    device: Optional[torch.device] = None,
) -> Trajectory:
    """Fixed-step pure-PyTorch integrator with piecewise jumps.

    Integrates from t=0 to t=T_end, applying jump updates at provided event times.
    Uses RK4 by default, falling back to Euler if dt is extremely small (to improve
    numerical stability in pathological step sizes).

    Parameters
    ----------
    ode_func:
        Callable(h, t) -> dh/dt. May close over parameters/modules. Should accept
        h: [batch, D], t: scalar 0-d tensor, and return [batch, D].
    h0:
        Initial state, shape [batch, D].
    event_times:
        1D strictly increasing tensor of event times within [0, T_end]. May be empty.
    jump_update:
        Optional callable(x_i, h_minus) -> delta h. If None, no jumps are applied.
    x_of_t:
        Optional callable(t) -> features [batch, X] for current time t (0-d tensor).
        Used to supply x_i to `jump_update` and can be used inside the ODE if captured.
        If None, `x_i` is treated as zeros (when jump_update expects inputs).
    dt:
        Fixed step size for the solver.
    T_end:
        Final time horizon. If None and `event_times` not empty, uses last event time.
        If None and `event_times` empty, raises ValueError.
    return_every_step:
        If True, record every solver step. Otherwise, record t=0, each event time
        (state just before jump), and T_end.
    device:
        Optional torch.device to place tensors. Defaults to h0.device.

    Returns
    -------
    Trajectory
        Container with `.times` and `.states` aligned.

    Notes
    -----
    - Determinism is controlled externally; this integrator is purely functional
      given `dt`, `ode_func`, and initial conditions.
    - Event handling: we record h(t_i^-) at event times, then apply jump to get h(t_i^+).
      We do not record post-jump states unless `return_every_step=True` implies
      recording at the next solver step.
    """
    if h0.ndim != 2:
        raise ValueError("h0 must have shape [batch, D]")

    dev = device if device is not None else h0.device
    dtype = h0.dtype

    # Normalize and validate event_times
    if event_times is None:
        event_times = torch.empty(0, device=dev, dtype=dtype)
    else:
        event_times = event_times.to(device=dev, dtype=dtype)
        if event_times.ndim != 1:
            raise ValueError("event_times must be a 1D tensor")
        if event_times.numel() > 1:
            if not torch.all(event_times[1:] > event_times[:-1]):
                raise ValueError("event_times must be strictly increasing")

    # Determine T_end
    if T_end is None:
        if event_times.numel() == 0:
            raise ValueError("T_end must be provided when event_times is empty.")
        T_end = float(event_times[-1].item())
    else:
        T_end = float(T_end)

    if dt <= 0.0:
        raise ValueError("dt must be positive")

    h = h0.to(device=dev, dtype=dtype)
    t = torch.tensor(0.0, device=dev, dtype=dtype)

    times: list[Tensor] = [t.clone()]
    states: list[Tensor] = [h.clone()]

    # Solver selection: RK4 default, Euler fallback for extremely small dt
    dt_tensor = torch.tensor(float(dt), device=dev, dtype=dtype)
    use_euler = float(dt) <= 1e-6

    def euler_step(y: Tensor, tt: Tensor, hdt: Tensor) -> Tensor:
        return y + hdt * ode_func(y, tt)

    def rk4_step(y: Tensor, tt: Tensor, hdt: Tensor) -> Tensor:
        k1 = ode_func(y, tt)
        k2 = ode_func(y + 0.5 * hdt * k1, tt + 0.5 * hdt)
        k3 = ode_func(y + 0.5 * hdt * k2, tt + 0.5 * hdt)
        k4 = ode_func(y + hdt * k3, tt + hdt)
        return y + (hdt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def one_step(y: Tensor, tt: Tensor, step_size: Tensor) -> Tuple[Tensor, Tensor]:
        new_y = euler_step(y, tt, step_size) if use_euler else rk4_step(y, tt, step_size)
        new_t = tt + step_size
        return new_y, new_t

    # March to each event time, applying jumps
    idx = 0
    n_events = int(event_times.numel())

    def maybe_record(current_t: Tensor, current_h: Tensor) -> None:
        if times and torch.isclose(current_t, times[-1]):
            # Avoid duplicate time stamps due to numerical rounding
            return
        times.append(current_t.clone())
        states.append(current_h.clone())

    def drive_at(time_scalar: Tensor) -> Optional[Tensor]:
        return x_of_t(time_scalar) if x_of_t is not None else None

    # Integrate up to each event, recording the state just before the jump
    while idx < n_events:
        t_event = event_times[idx]
        # Step until we reach the event time
        while t < t_event:
            step = torch.minimum(dt_tensor, t_event - t)
            if float(step.item()) <= 1e-12:
                # Snap to event time to avoid infinite loops due to rounding
                t = t_event
                break
            h, t = one_step(h, t, step)
            if return_every_step:
                maybe_record(t, h)

        # Record h(t_i^-) at the event
        maybe_record(t_event, h)

        # Apply jump update: h^+ = h^- + U(x_i, h^-)
        if jump_update is not None:
            x_i = drive_at(t_event)
            delta = jump_update(x_i, h)
            if delta.shape != h.shape:
                raise ValueError(f"jump_update must return shape {tuple(h.shape)}, got {tuple(delta.shape)}")
            h = h + delta

        idx += 1

    # March from last event (or 0) to T_end
    T_end_tensor = torch.tensor(T_end, device=dev, dtype=dtype)
    while t < T_end_tensor:
        step = torch.minimum(dt_tensor, T_end_tensor - t)
        if float(step.item()) <= 1e-12:
            t = T_end_tensor
            break
        h, t = one_step(h, t, step)
        if return_every_step:
            maybe_record(t, h)

    # Ensure final time/state recorded
    maybe_record(T_end_tensor, h)

    return Trajectory(times=torch.stack(times, dim=0), states=torch.stack(states, dim=0), marks=None)
