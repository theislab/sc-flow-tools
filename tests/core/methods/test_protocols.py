from typing import Any

import pytest
import torch

from sckitflow.core._types import PredictionData, StepData

# Assume the protocol module is importable; adjust the import as needed.
from sckitflow.core.methods._protocols import (
    BaseProtocol,
    InferenceProtocol,
    InferenceProtocolWrapper,
    Method,
    MethodWrapper,
    TrainingProtocol,
    TrainingProtocolWrapper,
    _AbstractInferenceProtocol,
    _AbstractMethod,
    _AbstractTrainingProtocol,
)


# -------------------- Dummy Implementations --------------------
class DummyModule(torch.nn.Module):
    """Simple module for testing; satisfies the BaseModule type informally."""

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 2)

    def forward(self, x):
        return self.linear(x)


class DummyStepData:
    """Minimal StepData stand-in."""

    pass


class DummyPredictionData:
    """Minimal PredictionData stand-in."""

    pass


# -------------------- Concrete Protocol Subclasses for Testing --------------------
class ConcreteTrainingProtocol(TrainingProtocol):
    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.tensor(0.0), {"loss": 0.0}


class ConcreteInferenceProtocol(InferenceProtocol):
    def predict(self, step_data: StepData) -> PredictionData:
        return DummyPredictionData()


class ConcreteMethod(Method):
    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.tensor(0.0), {"loss": 0.0}

    def predict(self, step_data: StepData) -> PredictionData:
        return DummyPredictionData()


# -------------------- Fixtures --------------------
@pytest.fixture
def dummy_module():
    return DummyModule()


@pytest.fixture
def step_data():
    return DummyStepData()


# -------------------- Abstractness Tests --------------------
@pytest.mark.parametrize(
    "cls",
    [
        _AbstractTrainingProtocol,
        _AbstractInferenceProtocol,
        _AbstractMethod,
        TrainingProtocol,
        InferenceProtocol,
        Method,
    ],
)
def test_abstract_classes_cannot_be_instantiated(cls, dummy_module):
    """Ensure abstract classes raise TypeError when instantiated directly."""
    with pytest.raises(TypeError):
        # All abstract classes require either module (for BaseProtocol subclasses)
        # or no arguments (for pure abstract contracts). We pass a dummy module
        # for those that accept it; for pure abstract, no args are needed.
        if issubclass(cls, BaseProtocol):
            cls(dummy_module)
        else:
            cls()


# -------------------- BaseProtocol Storage --------------------
def test_baseprotocol_initialization_and_properties(dummy_module):
    """Test BaseProtocol stores module, dtype, device and exposes them."""
    proto = BaseProtocol(dummy_module, dtype=torch.float64, device_id="cpu")
    assert proto.dtype == torch.float64
    assert proto.device_id == "cpu"
    assert proto.module is dummy_module
    # Check that the module was moved to the specified dtype
    assert next(proto.module.parameters()).dtype == torch.float64


def test_baseprotocol_set_train_mode(dummy_module):
    """Test set_train_mode toggles the underlying module's training flag."""
    proto = BaseProtocol(dummy_module)
    proto.set_train_mode(True)
    assert proto.module.training is True
    proto.set_train_mode(False)
    assert proto.module.training is False


# -------------------- Concrete Protocol Subclasses --------------------
def test_training_protocol_subclass(dummy_module, step_data):
    """Test a concrete TrainingProtocol subclass works and has properties."""
    proto = ConcreteTrainingProtocol(dummy_module, device_id="cpu")
    assert isinstance(proto, _AbstractTrainingProtocol)
    assert proto.dtype == torch.float32  # default
    assert proto.device_id == "cpu"
    loss, meta = proto.train_step(step_data)
    assert loss.item() == 0.0
    assert meta == {"loss": 0.0}


def test_inference_protocol_subclass(dummy_module, step_data):
    """Test a concrete InferenceProtocol subclass works and has properties."""
    proto = ConcreteInferenceProtocol(dummy_module, device_id="cpu")
    assert isinstance(proto, _AbstractInferenceProtocol)
    pred = proto.predict(step_data)
    assert isinstance(pred, DummyPredictionData)


def test_method_subclass(dummy_module, step_data):
    """Test a concrete Method subclass implements both methods and has properties."""
    proto = ConcreteMethod(dummy_module, device_id="cpu")
    assert isinstance(proto, _AbstractMethod)
    loss, meta = proto.train_step(step_data)
    assert loss.item() == 0.0
    pred = proto.predict(step_data)
    assert isinstance(pred, DummyPredictionData)
    assert proto.dtype == torch.float32
    assert proto.device_id == "cpu"


# -------------------- Wrapper Tests --------------------
def test_training_protocol_wrapper_delegates(dummy_module, step_data):
    """TrainingProtocolWrapper should delegate train_step and expose properties."""
    inner = ConcreteTrainingProtocol(dummy_module, dtype=torch.float64, device_id="cpu")
    wrapper = TrainingProtocolWrapper(inner)

    # Delegation
    loss, meta = wrapper.train_step(step_data)
    assert loss.item() == 0.0
    assert meta == {"loss": 0.0}

    # Property delegation
    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module


def test_inference_protocol_wrapper_delegates(dummy_module, step_data):
    """InferenceProtocolWrapper should delegate predict and expose properties."""
    inner = ConcreteInferenceProtocol(dummy_module, dtype=torch.float64, device_id="cpu")
    wrapper = InferenceProtocolWrapper(inner)

    pred = wrapper.predict(step_data)
    assert isinstance(pred, DummyPredictionData)

    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module


def test_method_wrapper_delegates_both(dummy_module, step_data):
    """MethodWrapper should delegate both train_step and predict."""
    inner = ConcreteMethod(dummy_module, dtype=torch.float64, device_id="cpu")
    wrapper = MethodWrapper(inner)

    loss, meta = wrapper.train_step(step_data)
    assert loss.item() == 0.0
    pred = wrapper.predict(step_data)
    assert isinstance(pred, DummyPredictionData)

    assert wrapper.dtype == torch.float64
    assert wrapper.device_id == "cpu"
    assert wrapper.module is dummy_module


# -------------------- Type Checking in Wrappers --------------------
def test_training_wrapper_rejects_non_training_protocol(dummy_module):
    """TrainingProtocolWrapper should raise TypeError for invalid wrapped object."""
    # Use an object that is not a training protocol (e.g., BaseProtocol alone)
    with pytest.raises(TypeError):
        TrainingProtocolWrapper(BaseProtocol(dummy_module))


def test_inference_wrapper_rejects_non_inference_protocol(dummy_module):
    """InferenceProtocolWrapper should raise TypeError for invalid wrapped object."""
    with pytest.raises(TypeError):
        InferenceProtocolWrapper(BaseProtocol(dummy_module))


def test_method_wrapper_rejects_non_method_protocol(dummy_module):
    """MethodWrapper should raise TypeError for wrapped object that is not _AbstractMethod."""
    with pytest.raises(TypeError):
        # ConcreteTrainingProtocol is only training, not full method
        MethodWrapper(ConcreteTrainingProtocol(dummy_module))


# -------------------- Property Delegation Errors --------------------
class BareTrainingProtocol(_AbstractTrainingProtocol):
    """Implements train_step but has no dtype/device/module properties."""

    def train_step(self, step_data: StepData) -> tuple[torch.Tensor, dict[str, Any]]:
        return torch.tensor(0.0), {}


def test_wrapper_property_attribute_error_when_missing():
    """Wrapper properties should raise AttributeError if wrapped protocol lacks them."""
    bare = BareTrainingProtocol()
    wrapper = TrainingProtocolWrapper(bare)

    # train_step works
    loss, meta = wrapper.train_step(DummyStepData())
    assert loss.item() == 0.0

    # properties should raise AttributeError
    with pytest.raises(AttributeError):
        _ = wrapper.dtype
    with pytest.raises(AttributeError):
        _ = wrapper.device_id
    with pytest.raises(AttributeError):
        _ = wrapper.module
