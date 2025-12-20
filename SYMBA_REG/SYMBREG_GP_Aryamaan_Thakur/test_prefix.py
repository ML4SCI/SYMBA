import unittest
import sympy as sp
from utils.sympy_prefix import sympy_to_prefix

class TestSympyPrefix(unittest.TestCase):
    def setUp(self):
        self.x, self.y = sp.symbols('x y')

    def test_simple_arithmetic(self):
        # Test Addition: x + y -> ['add', 'x', 'y']
        expr = self.x + self.y
        expected = ['add', 'x', 'y']
        self.assertEqual(sympy_to_prefix(expr), expected)

        # Test Multiplication: x * y -> ['mul', 'x', 'y']
        expr = self.x * self.y
        expected = ['mul', 'x', 'y']
        self.assertEqual(sympy_to_prefix(expr), expected)
        
        # Test Subtraction: x-y -> ['add', 'x', 'mul', 's-', '1', 'y'] 
        # Note: SymPy represents x-y as x + (-1)*y
        expr = self.x - self.y
        prefix = sympy_to_prefix(expr)
        self.assertTrue(isinstance(prefix, list))
        self.assertIn('add', prefix)
        self.assertIn('x', prefix)

    def test_power_functions(self):
        # Test Power: x**2
        expr = self.x**2
        # Expected : ['pow', 'x', 's+', '2'] or similar based on format_integer
        prefix = sympy_to_prefix(expr)
        self.assertEqual(prefix[0], 'pow')
        self.assertEqual(prefix[1], 'x')
        # Check integer formatting
        self.assertIn('2', prefix)

    def test_nested_expressions(self):
        # Test sin(x + y)
        expr = sp.sin(self.x + self.y)
        expected = ['sin', 'add', 'x', 'y']
        self.assertEqual(sympy_to_prefix(expr), expected)

        # Test exp(x * y)
        expr = sp.exp(self.x * self.y)
        expected = ['exp', 'mul', 'x', 'y']
        self.assertEqual(sympy_to_prefix(expr), expected)

    def test_integer_formatting(self):
        # Test how integers are formated
        expr = sp.Integer(42)
        # sympy_prefix.format_integer usage: ['s+', '4', '2']
        prefix = sympy_to_prefix(expr)
        self.assertEqual(prefix, ['s+', '4', '2'])

if __name__ == '__main__':
    unittest.main()
