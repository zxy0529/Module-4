"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable


#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    return x * y


def id(x: float) -> float:
    return x


def add(x: float, y: float) -> float:
    return x + y


def neg(x: float) -> float:
    return -x


def lt(x: float, y: float) -> bool:
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> bool:
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    return 1.0 if math.fabs(x - y) < 1e-2 else 0.0


def sigmoid(x:float)->float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x:float)->float:
    return x if x > 0 else 0.0


def log(x:float)->float:
    return math.log(x)


def exp(x:float)->float:
    return math.exp(x)


def inv(x:float)->float:
    return 1.0 / x


def log_back(x:float, d:float)->float:
    return d / x


def inv_back(x:float, d:float)->float:
    return -d / (x ** 2)


def relu_back(x:float, d:float)->float:
    return d if x > 0 else 0.0

# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


# TODO: Implement for Task 0.3.
def map(fn):
    def apply(ls):
        return [fn(t) for t in ls]
    return apply

def zipWith(fn):
    def apply(ls1,ls2):
        arr = []
        for i in range(len(ls1)):
            arr.append(fn(ls1[i],ls2[i]))
        return arr
    return apply

def reduce(fn, start):
    def apply(ls):
        res = start
        for t in ls:
            res = fn(res,t)
        return res
    return apply

def negList(ls:Iterable[float])->Iterable[float]:
    return map(neg)(ls)

def addLists(ls1:Iterable[float], ls2:Iterable[float])->Iterable[float]:
    return zipWith(add)(ls1, ls2)

def sum(ls:Iterable[float])->float:
    return reduce(add, 0)(ls)

def prod(ls:Iterable[float])->float:
    return reduce(mul, 1)(ls)

