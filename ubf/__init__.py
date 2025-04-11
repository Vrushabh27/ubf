"""
Universal Barrier Function (UBF) - A framework for implementing UBF methods.

This package serves as a universal framework for implementing UBF methods
with modular design to allow users to easily replace all tuning parameters,
dynamics, and safety constraints.
"""

__version__ = '0.1.0'
__author__ = 'Vrushabh Zinage, Efstathios Bakolas'

from ubf.core import ubf_core
from ubf.numerics import jacobian, integration 