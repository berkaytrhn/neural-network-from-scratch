import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

class Init(ABC):
    @abstractmethod
    def initialize(self, shape, seed):
        """
        Base Class Abstract Method For Initialization
        """
        pass

class XavierInit(Init):
    def initialize(self, shape: tuple, seed: int) -> None:
        in_dim, out_dim = shape
        key=None
        scale = jnp.sqrt(2.0 / (in_dim + out_dim))
        return scale * jax.random.normal(key, shape)

