import abc
from typing import Any

import torch

from sckitflow.core._types import PredictionData, StepData
from sckitflow.core.nn._modules import BaseModule

__all__ = [
    "_AbstractTrainingProtocol",
    "_AbstractInferenceProtocol",
    "_AbstractMethod",
    "BaseProtocol",
    "TrainingProtocol",
    "InferenceProtocol",
    "Method",
    "TrainingProtocolMixin",
    "InferenceProtocolMixin",
    "TrainingProtocolWrapper",
    "InferenceProtocolWrapper",
    "MethodWrapper",
]


# -------------------- Abstract Contracts (no storage) --------------------
class _AbstractTrainingProtocol(abc.ABC):
    """Pure abstract contract for training protocols."""

    @abc.abstractmethod
    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]: ...


class _AbstractInferenceProtocol(abc.ABC):
    """Pure abstract contract for inference protocols."""

    @abc.abstractmethod
    def predict(self, step_data: StepData) -> PredictionData: ...


class _AbstractMethod(_AbstractTrainingProtocol, _AbstractInferenceProtocol):
    """Combined contract for full training + inference protocols."""

    pass


# -------------------- Storage Base --------------------
class BaseProtocol:
    """Mixin providing storage for module, dtype, and device.

    This class does **not** inherit from any abstract protocol; it only holds
    the neural module and associated properties. Concrete protocol classes
    combine this with the appropriate abstract contracts.
    """

    def __init__(
        self,
        module: BaseModule,
        dtype: torch.dtype = torch.float32,
        device_id: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        self._dtype = dtype
        self._device_id = device_id
        self._module = module.to(self._dtype).to(self._device_id)

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device_id(self) -> str:
        return self._device_id

    @property
    def module(self) -> BaseModule:
        return self._module

    def set_train_mode(self, mode: bool) -> None:
        """Set the underlying module to training or evaluation mode."""
        if mode:
            self.module.train()
        else:
            self.module.eval()


# -------------------- Base Protocol Classes (still abstract) --------------------
class TrainingProtocol(BaseProtocol, _AbstractTrainingProtocol):
    """Base class for training‑only protocols.

    Subclass this and implement `train_step`. The module, dtype, and device are
    stored automatically.
    """

    pass


class InferenceProtocol(BaseProtocol, _AbstractInferenceProtocol):
    """Base class for inference‑only protocols.

    Subclass this and implement `predict`. The module, dtype, and device are
    stored automatically.
    """

    pass


class Method(BaseProtocol, _AbstractMethod):
    """Base class for full protocols (training + inference).

    Subclass this and implement both `train_step` and `predict`.
    """

    pass


# -------------------- Common Wrapper Mixin Base (internal) --------------------
class _BaseProtocolMixin:
    """Shared delegation logic for wrapper mixins."""

    def __init__(self, protocol):
        self._protocol = protocol

    @property
    def protocol(self):
        return self._protocol

    @property
    def dtype(self) -> torch.dtype:
        if hasattr(self._protocol, "dtype"):
            return self._protocol.dtype
        raise AttributeError("Wrapped protocol does not expose 'dtype'.")

    @property
    def device_id(self) -> str:
        if hasattr(self._protocol, "device_id"):
            return self._protocol.device_id
        raise AttributeError("Wrapped protocol does not expose 'device_id'.")

    @property
    def module(self) -> BaseModule:
        if hasattr(self._protocol, "module"):
            return self._protocol.module
        raise AttributeError("Wrapped protocol does not expose 'module'.")


# -------------------- Specific Wrapper Mixins --------------------
class TrainingProtocolMixin(_BaseProtocolMixin):
    """Mixin for wrappers that wrap a training protocol."""

    pass


class InferenceProtocolMixin(_BaseProtocolMixin):
    """Mixin for wrappers that wrap an inference protocol."""

    pass


# -------------------- Ready‑made Wrapper Classes --------------------
class TrainingProtocolWrapper(TrainingProtocolMixin, _AbstractTrainingProtocol):
    """Concrete wrapper for a training protocol.

    Delegates `train_step` to the wrapped protocol.
    """

    def __init__(self, protocol):
        if not isinstance(protocol, _AbstractTrainingProtocol):
            raise TypeError("Wrapped protocol must provide train_step.")
        super().__init__(protocol)

    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._protocol.train_step(step_data)


class InferenceProtocolWrapper(InferenceProtocolMixin, _AbstractInferenceProtocol):
    """Concrete wrapper for an inference protocol.

    Delegates `predict` to the wrapped protocol.
    """

    def __init__(self, protocol):
        if not isinstance(protocol, _AbstractInferenceProtocol):
            raise TypeError("Wrapped protocol must provide predict.")
        super().__init__(protocol)

    def predict(self, step_data: StepData) -> PredictionData:
        return self._protocol.predict(step_data)


class MethodWrapper(TrainingProtocolMixin, InferenceProtocolMixin, _AbstractMethod):
    """Concrete wrapper for a full protocol (training + inference).

    Delegates both `train_step` and `predict` to the wrapped protocol.
    """

    def __init__(self, protocol):
        if not isinstance(protocol, _AbstractMethod):
            raise TypeError("Wrapped protocol must provide both train_step and predict.")
        # Initialize both mixins; they share the same protocol reference.
        TrainingProtocolMixin.__init__(self, protocol)
        InferenceProtocolMixin.__init__(self, protocol)

    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return self._protocol.train_step(step_data)

    def predict(self, step_data: StepData) -> PredictionData:
        return self._protocol.predict(step_data)
