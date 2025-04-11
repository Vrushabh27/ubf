"""
Unit tests for the core UBF functionality.
"""

import unittest
import numpy as np
from ubf.core.ubf_core import log_sum_exp, construct_ubf
from ubf.numerics.jacobian import numerical_jacobian
from ubf.numerics.integration import forward_euler_integration


class TestUBFCore(unittest.TestCase):
    """Test cases for the UBF core functionality."""
    
    def test_log_sum_exp(self):
        """Test the log-sum-exp function for combining barrier functions."""
        # Define simple barrier functions
        def h1(x, u):
            return x[0]**2 + x[1]**2 - 1.0
            
        def h2(x, u):
            return x[0]**2 - 0.5
            
        h_funcs = [h1, h2]
        x = np.array([0.5, 0.5])
        u = np.array([0.0])
        beta = 10.0
        
        # Compute the expected results manually
        h_vals = np.array([beta * h(x, u) for h in h_funcs])
        h_min = np.min(h_vals)
        sum_exp = np.sum(np.exp(-(h_vals - h_min)))
        expected = (h_min + np.log(sum_exp) + np.log(1/len(h_funcs))) / beta
        
        # Compute the actual result
        result = log_sum_exp(beta, h_funcs, x, u) / beta
        
        self.assertAlmostEqual(result, expected, places=6)
        
    def test_construct_ubf(self):
        """Test the construction of a UBF from individual barrier functions."""
        # Define simple barrier functions
        def h1(x, u):
            return x[0]**2 + x[1]**2 - 1.0
            
        def h2(x, u):
            return x[0]**2 - 0.5
            
        h_funcs = [h1, h2]
        x = np.array([0.5, 0.5])
        u = np.array([0.0])
        beta = 10.0
        
        # Construct the UBF
        h_ubf = construct_ubf(h_funcs, beta)
        
        # Compute the expected result manually
        h_vals = np.array([beta * h(x, u) for h in h_funcs])
        h_min = np.min(h_vals)
        sum_exp = np.sum(np.exp(-(h_vals - h_min)))
        expected = (h_min + np.log(sum_exp) + np.log(1/len(h_funcs))) / beta
        
        # Compute the actual result
        result = h_ubf(x, u)
        
        self.assertAlmostEqual(result, expected, places=6)
        
    def test_numerical_jacobian(self):
        """Test the numerical Jacobian computation."""
        def f(x):
            return np.array([x[0]**2 + x[1], x[0] * x[1]])
            
        x = np.array([2.0, 3.0])
        
        # Compute numerical Jacobian
        J = numerical_jacobian(f, x, 2)
        
        # Expected Jacobian (computed analytically)
        expected_J = np.array([
            [2*x[0], 1.0],
            [x[1], x[0]]
        ])
        
        # Check if they are close
        np.testing.assert_array_almost_equal(J, expected_J, decimal=5)
        
    def test_forward_euler(self):
        """Test the forward Euler integration."""
        def f(x, u):
            return np.array([x[1], u[0] - 0.1 * x[1] - x[0]])
            
        x0 = np.array([1.0, 0.0])  # Initial position and velocity
        u = np.array([0.0])        # Zero input force
        dt = 0.01
        steps = 100
        
        # Integrate using forward Euler
        x_final = forward_euler_integration(x0, u, f, steps, dt)
        
        # For simple harmonic oscillator with damping and no input,
        # the position should decrease in magnitude
        self.assertLess(abs(x_final[0]), abs(x0[0]))


if __name__ == "__main__":
    unittest.main() 