# https://github.com/fastai/fastai/blob/master/fastai/imports.py
from typing import Union,Optional,Dict,List,Tuple,Sequence,Mapping,Callable,Iterable,Any,NamedTuple
import io,operator,sys,os,re,mimetypes,csv,itertools,json,shutil,glob,pickle,tarfile,collections
import hashlib,itertools,types,inspect,functools,time,math,bz2,typing,numbers,string
import multiprocessing,threading,urllib,tempfile,concurrent.futures,warnings,zipfile
import numpy as np,pandas as pd,scipy

# jax related
import jax
import jax.numpy as jnp, jax.random as jrand, jax.scipy as jsp

# nn related
import haiku as hk
import optax
import chex

# misc
from pydantic import BaseModel as BaseConfig, validator, ValidationError, Field
from pathlib import Path
from abc import ABC, abstractmethod

# jax warnings
warnings.simplefilter(action='ignore', category=FutureWarning)    
