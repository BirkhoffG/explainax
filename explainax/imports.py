# https://github.com/fastai/fastai/blob/master/fastai/imports.py
from __future__ import annotations, print_function, division
from typing import Union,Optional,Dict,List,Tuple,Sequence,Mapping,Callable,Iterable,Any,NamedTuple
import io,operator,sys,os,re,mimetypes,csv,itertools,json,shutil,glob,pickle,tarfile,collections
import hashlib,itertools,types,inspect,functools,time,math,bz2,typing,numbers,string
import multiprocessing,threading,urllib,tempfile,concurrent.futures,warnings,zipfile
import numpy as np

# jax related
import jax
import jax.numpy as jnp, jax.random as jrand, jax.scipy as jsp
from jax import jit, vmap, device_put, lax, grad, value_and_grad, Array

# nn related
import optax
import chex

# misc
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import partial

# jax warnings
warnings.simplefilter(action='ignore', category=FutureWarning)    


@dataclass
class Config:
    """Global configuration for the library"""
    rng_reserve_size: int
    global_seed: int

    @classmethod
    def default(cls) -> Config:
        return cls(rng_reserve_size=1, global_seed=42)


def get_config() -> Config:
    """Get the global configuration for the library"""
    return Config.default()
