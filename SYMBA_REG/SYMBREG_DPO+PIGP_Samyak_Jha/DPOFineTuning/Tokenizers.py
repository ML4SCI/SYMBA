import pandas as pd
import numpy as np
import sympy as sp

variables = [
        's_1',
        's_2',
        's_3',
        's_4',
        's_5',
        's_6',
        's_7',
        's_8',
        's_9',
        's_10',
        's_11'
        ]

operators = {
    # Elementary functions
    sp.Add: 'add',
    sp.Mul: 'mul',
    sp.Pow: 'pow',
    sp.exp: 'exp',
    sp.log: 'ln',
    sp.Abs: 'abs',
    sp.sign: 'sign',
#     sp.Sub: 'sub',
#     sp.Div: 'div',
    # Trigonometric Functions
    sp.sin: 'sin',
    sp.cos: 'cos',
    sp.tan: 'tan',
    sp.cot: 'cot',
    sp.sec: 'sec',
    sp.csc: 'csc',
    # Trigonometric Inverses
    sp.asin: 'asin',
    sp.acos: 'acos',
    sp.atan: 'atan',
    sp.acot: 'acot',
    sp.asec: 'asec',
    sp.acsc: 'acsc',
    # Hyperbolic Functions
    sp.sinh: 'sinh',
    sp.cosh: 'cosh',
    sp.tanh: 'tanh',
    sp.coth: 'coth',
    sp.sech: 'sech',
    sp.csch: 'csch',
    # Hyperbolic Inverses
    sp.asinh: 'asinh',
    sp.acosh: 'acosh',
    sp.atanh: 'atanh',
    sp.acoth: 'acoth',
    sp.asech: 'asech',
    sp.acsch: 'acsch',
    sp.Min: 'min',
    sp.Max: 'max',
    # Derivative
    sp.Derivative: 'derivative',
}

operators_inv = {operators[key]: key for key in operators}
operators_inv.update({'sub': lambda x, y: x - y,'div': lambda x, y: x / y})
operators_inv["mul("] = sp.Mul
operators_inv["add("] = sp.Add

operators_nargs = {
    # Elementary functions
    'mul(': -1,
    'add(': -1,
    'add': 2,
    'sub': 2,
    'mul': 2,
    'div': 2,
    'pow': 2,
    'rac': 2,
    'inv': 1,
    'pow2': 1,
    'pow3': 1,
    'pow4': 1,
    'pow5': 1,
    'sqrt': 1,
    'exp': 1,
    'ln': 1,
    'abs': 1,
    'sign': 1,
    'max': 2,
    'min': 2,
    # Trigonometric Functions
    'sin': 1,
    'cos': 1,
    'tan': 1,
    'cot': 1,
    'sec': 1,
    'csc': 1,
    # Trigonometric Inverses
    'asin': 1,
    'acos': 1,
    'atan': 1,
    'acot': 1,
    'asec': 1,
    'acsc': 1,
    # Hyperbolic Functions
    'sinh': 1,
    'cosh': 1,
    'tanh': 1,
    'coth': 1,
    'sech': 1,
    'csch': 1,
    # Hyperbolic Inverses
    'asinh': 1,
    'acosh': 1,
    'atanh': 1,
    'acoth': 1,
    'asech': 1,
    'acsch': 1,
    # Derivative
    'derivative': 2,
    # custom functions
    'f': 1,
    'g': 2,
    'h': 3,
}

masses_strings = [
        "m_e",
        "m_u",
        "m_d",
        "m_s",
        "m_c",
        "m_b",
        "m_t",
        ]

masses = [sp.Symbol(x) for x in masses_strings]

# these will be converted to the numbers format in `format_number`
integers_types = [
        sp.core.numbers.Integer,
        sp.core.numbers.One,
        sp.core.numbers.NegativeOne,
        sp.core.numbers.Zero,
        ]

numbers_types = integers_types + [
    sp.core.numbers.Rational,
    sp.core.numbers.Half,
    sp.core.numbers.Exp1,
    sp.core.numbers.Pi,
    sp.core.numbers.ImaginaryUnit,
    sp.core.numbers.Float,
]

# don't continue evaluating at these, but stop
atoms = [
    str,
    sp.core.symbol.Symbol,
    sp.core.numbers.Exp1,
    sp.core.numbers.Pi,
    sp.core.numbers.ImaginaryUnit,
] + numbers_types


Inverse_trig = {
    'arcsin': 'asin',
    'arccos': 'acos',
    'arctan': 'atan',
    'arccot': 'acot',
    'arcsec': 'asec',
    'arccsc': 'acsc',
    'arcsinh': 'asinh',
    'arccosh': 'acosh',
    'arctanh': 'atanh',
    'arccoth': 'acoth',
    'arcsech': 'asech',
    'arccsch': 'acsch',         
}

functional_to_sympy = {
    'add': sp.Add,
    'sub': lambda x, y: sp.Add(x, -y),
    'mul': sp.Mul,
    'div': lambda x, y: sp.Mul(x, sp.Pow(y, -1)),
    'pow': sp.Pow,
    'sin': sp.sin,
    'cos': sp.cos,
    'tan': sp.tan,
    'abs': sp.Abs,
    'max': sp.Max,
    'min': sp.Min,
    'tanh': sp.tanh,
    'protected_div': lambda x, y: sp.Mul(x, sp.Pow(y, -1)) if y != 0 else 1,
    'protected_pow': sp.Pow,
    'protected_exp': sp.exp,
    'protected_log': lambda x : sp.log(sp.Abs(x)),
    'protected_sqrt': lambda x : sp.sqrt(sp.Abs(x)),
    'pi': sp.pi,
    'neg': lambda x: -x
}

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

class Tokenizer:
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.word2id = {}
        self.id2word = {}

        with open(vocab_path) as file:
            words = map(lambda x: x.rstrip('\n'), file.readlines())

        for (n, word) in enumerate(words):
            self.word2id[word] = n
            self.id2word[n] = word 

    def encode(self, lst):
        return np.array([[self.word2id[j] for j in i] for i in lst], dtype=np.ushort)

    def decode(self, lst):
        return [[self.id2word[j] for j in i] for i in lst]
    
class Encoder_tokeniser(Tokenizer):
    def __init__(self,float_precision,mantissa_len,max_exponent,vocab_path,max_len = 10):
        super().__init__(vocab_path)
        
        self.max_len = max_len
        self.float_precision = float_precision
        self.mantissa_len = mantissa_len
        self.max_exponent = max_exponent
        self.base = (self.float_precision + 1) // self.mantissa_len
        self.max_token = 10 ** self.base
        
    def pre_tokenize(self, data):
        arr = np.array([i.split() for i in data], dtype=np.float32)
        permutation = [-1] + [i for i in range(arr.shape[1]-1)]
        arr = np.pad(arr[:, permutation], ((0,0), (0, self.max_len - arr.shape[1])), mode="constant", constant_values=[-np.inf])
        return arr
    
    def tokenize(self, data):
        out = self.pre_tokenize(data)
        out = self.encode_float(out)
        out = self.encode(out)
        return out
        
    def encode_float(self,values):
        if len(values.shape) == 1:
            seq = []
            value = values
            for val in value:
                if val in [-np.inf, np.inf]:
                    seq.extend(['<pad>']*3)
                    continue
                
                sign = "+" if val >= 0 else "-"
                m, e = (f"%.{self.float_precision}e" % val).split("e")
                i, f = m.lstrip("-").split(".")
                i = i + f
                tokens = chunks(i, self.base)
                expon = int(e) - self.float_precision
                if expon < -self.max_exponent:
                    tokens = ["0" * self.base] * self.mantissa_len
                    expon = int(0)
                seq.extend([sign, *["N" + token for token in tokens], "E" + str(expon)])
            return seq
        else:
            seqs = [self.encode_float(values[0])]
            N = values.shape[0]
            for n in range(1, N):
                seqs += [self.encode_float(values[n])]
        return seqs
    def decode_float(self,seq):
        if len(seq) == 0:
            return None
        decoded_seq = []
        for val in chunks(seq, 2 + self.mantissa_len):
            for x in val:
                if x[0] not in ["-", "+", "E", "N"]:
                    return np.nan
            try:
                sign = 1 if val[0] == "+" else -1
                mant = ""
                for x in val[1:-1]:
                    mant += x[1:]
                mant = int(mant)
                exp = int(val[-1][1:])
                value = sign * mant * (10 ** exp)
                value = float(value)
            except Exception:
                value = np.nan
            decoded_seq.append(value)
        return decoded_seq

def flatten(l, ltypes=(list, tuple)):
    """
    flatten a python list
    from http://rightfootin.blogspot.com/2006/09/more-on-python-flatten.html
    """
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)

def sympy_to_prefix(expression):
    """
    Recursively go from a sympy expression to a prefix notation.
    Returns a flat list of tokens.
    """
    return flatten(sympy_to_prefix_rec(expression, []))

def sympy_to_prefix_rec(expression, ret):
    """
    Recursively go from a sympy expression to a prefix notation.
    The operators all get converted to their names in the array `operators`.
    Returns a nested list, where the nesting basically stands for parentheses.
    Since in prefix notation with a fixed number of arguments for each function (given in `operators_nargs`),
    parentheses are not needed, we can flatten the list later.
    """
    if expression in [sp.core.numbers.Pi, sp.core.numbers.ImaginaryUnit]:
        f = expression
    else:
        f = expression.func
    if f in atoms:
        if type(expression) in numbers_types:
            return ret + format_number(expression)
        return ret + [str(expression)]
    f_str = operators[f]
    f_nargs = operators_nargs[f_str]
    args = expression.args
    if len(args) == 1 & f_nargs == 1:
        ret = ret + [f_str]
        return sympy_to_prefix_rec(args[0], ret)
    if len(args) == 2:
        ret = ret + [f_str, sympy_to_prefix_rec(args[0], []), sympy_to_prefix_rec(args[1], [])]
    if len(args) > 2:
        args = list(map(lambda x: sympy_to_prefix_rec(x, []), args))
        ret = ret + repeat_operator_until_correct_binary(f_str, args)
    return ret

def repeat_operator_until_correct_binary(op, args, ret=[]):
    """
    sympy is not strict enough with the number of arguments.
    E.g. multiply takes a variable number of arguments, but for
    prefix notation it needs to ALWAYS have exactly 2 arguments

    This function is only for binary operators.

    Here I choose the convention as follows:
        1 + 2 + 3 --> + 1 + 2 3

    This is the same convention as in https://arxiv.org/pdf/1912.01412.pdf
    on page 15.

    input:
        op: in string form as in the list `operators`
        args: [arg1, arg2, ...] arguments of the operator, e.c. [1, 2, x**2,
                ...]. They can have other things to be evaluated in them
        ret: the list you already have. Usually []. Watch out, I think one has to explicitely give [],
            otherwise somehow the default value gets mutated, which I find a strange python behavior.
    """

    is_binary = operators_nargs[op] == 2
    assert is_binary, "repeat_operator_until_correct_binary only takes binary operators"

    if len(args) == 0:
        return ret
    elif len(ret) == 0:
        ret = [op] + args[-2:]
        args = args[:-2]
    else:
        ret = [op] + args[-1:] + ret
        args = args[:-1]

    return repeat_operator_until_correct_binary(op, args, ret)

def format_number(number):
    if type(number) in integers_types:
        return format_integer(number)
    elif type(number) == sp.core.numbers.Rational or type(number) == sp.core.numbers.Float:
        return format_rational(sp.Rational(number))
    elif type(number) == sp.core.numbers.Half:
        return format_half()
    elif type(number) == sp.core.numbers.Exp1:
        return format_exp1()
    elif type(number) == sp.core.numbers.Pi:
        return format_pi()
    elif type(number) == sp.core.numbers.ImaginaryUnit:
        return format_imaginary_unit()
    else:
        raise NotImplementedError
        
def format_float(number):
    return [str(number)]

def format_exp1():
    return ['E']

def format_pi():
    return ['pi']

def format_imaginary_unit():
    return ['I']

def format_half():
    """
    for some reason in sympy 1/2 is its own object and not a rational.
    This function formats it correctly like `format_rational`
    """
    return ['mul'] + ['s+', '1'] + ['pow'] + ['s+', '2'] + ["s-", "1"]

def format_rational(number):
    # for some reason number.p is a string
    p = sp.sympify(number.p)
    q = sp.sympify(number.q)
    return ['mul'] + format_integer(p) + ['pow'] + format_integer(q) + ['s-', '1']

def format_integer(integer):
    """take a sympy integer and format it as in
    https://arxiv.org/pdf/1912.01412.pdf

    input:
        integer: a `sympy.Integer` object, e.g. `sympy.Integer(-1)`

    output:
        [sign_token, digit0, digit1, ...]
        where sign_token is 's+' or 's-'

    Example:
        format_integer(sympy.Integer(-123))
        >> ['s-', '1', '2', '3']

    Implementation notes:
    Somehow Integer inherits from Rational in Sympy and a rational is p/q,
    so integer.p is used to extract the number.
    """
    # plus_sign = "s+"
    plus_sign = "s+"
    minus_sign = "s-"
    abs_num = abs(integer.p)
    is_neg = integer.could_extract_minus_sign()
    digits = list(str(abs_num))
    # digits = [str(abs_num)]

    if is_neg:
        ret = [minus_sign] + digits
    else:
        ret = [plus_sign] + digits

    return ret

def parse_if_str(x):
    if isinstance(x, str):
        return sp.parsing.parse_expr(x)
    return x

def rightmost_string_pos(expr_arr, pos=-1):
    if isinstance(expr_arr[pos], str):
        return len(expr_arr)+pos
    else:
        return rightmost_string_pos(expr_arr, pos-1)


def rightmost_operand_pos(expr, pos):
    operators = list(operators_inv.keys()) + ["s+", "s-"] + variables
    if expr[pos] in operators:
        return pos
    else:
        return rightmost_operand_pos(expr, pos-1)

def unformat_integer(arr):
    """
    inverse of the function format_integer.

    input:
        arr: array of strings just as the output of format_integer. E.g. ["s+", "4", "2"]

    output:
        the correspinding sympy integer, e.g. sympy.Integer(42) in the above example.

    The sign tokens are "s+" for positive integers and "s-" for negative. 0 comes with "s+", but does not matter.

    """
    sign_token = arr[0]
    ret = "-" if sign_token == "s-" else ""
    for s in arr[1:]:
        ret += str(s)

    return sp.parsing.parse_expr(ret)

def prefix_to_sympy(expr_arr):
    if len(expr_arr) == 1:
        return parse_if_str(expr_arr[0])
    op_pos = rightmost_operand_pos(expr_arr,len(expr_arr) - 1)
    if (op_pos == -1) | (op_pos == len(expr_arr)):
        print("something went wrong, operator should not be at end of array")
    op = expr_arr[op_pos]
    if op in operators_inv.keys():
        num_args = operators_nargs[op]
        op = operators_inv[op]
        args = expr_arr[op_pos+1:op_pos+num_args+1]
        args = [parse_if_str(a) for a in args]
        func = op(*args)
        expr = expr_arr[0:op_pos] + [func] + expr_arr[op_pos+num_args+1:]
        return prefix_to_sympy(expr)

    elif (op == 's+') | (op == "s-"):
        # int_end_pos = rightmost_int_pos(expr_arr)
        string_end_pos = rightmost_string_pos(expr_arr)
        integer = unformat_integer(expr_arr[op_pos:string_end_pos+1])
        expr_arr_new = expr_arr[0:op_pos] + [integer] + expr_arr[string_end_pos+1:]
        return prefix_to_sympy(expr_arr_new)
    elif op in variables:
        op = sp.sympify(op)
        expr_arr_new = expr_arr[0:op_pos] + [op] + expr_arr[op_pos+1:]
        return prefix_to_sympy(expr_arr_new)

    return op

class DecoderTokenizer(Tokenizer):
    def __init__(self, vocab_path):
        super().__init__(vocab_path)

    def equation_encoder(self, data):
        return [sympy_to_prefix(expr) for expr in data]
    
    def equation_decoder(self, data):
        return [prefix_to_sympy(lst) for lst in data]

    def pre_tokenize(self, data):
        return data
    
    def tokenize(self, data):
        out = self.pre_tokenize(data)
        out = self.equation_encoder(out)
        out = [['<bos>'] + i + ['<eos>'] for i in out]
        out = self.encode(out)
        return out
    
    def reverse_tokenize(self, data):
        out = self.decode(data)
        out = self.equation_decoder(out)
        return out

def convert_to_functional_form(expr):
    # Ensure the input is a SymPy expression
    expr = sp.sympify(expr)
    
    # Handle operators (e.g., Add, Mul, Pow)
    for key, value in operators.items():
        if isinstance(expr, key):
            args = expr.args
            
            # Handle binary operations with more than two arguments (e.g., nested operations)
            if key in [sp.Add, sp.Mul, sp.Pow] and len(args) > 2:
                result = f"{value}({convert_to_functional_form(args[0])}, {convert_to_functional_form(args[1])})"
                for arg in args[2:]:
                    result = f"{value}({result}, {convert_to_functional_form(arg)})"
                return result
            
            # For regular operations or binary with exactly 2 arguments
            return f"{value}({', '.join(convert_to_functional_form(arg) for arg in args)})"
    
    # Handle rational numbers
    if isinstance(expr, sp.Rational):
        # Skip the conversion if it's effectively an integer (denominator is 1)
        if expr.q == 1:
            return str(expr.p)
        return f"div({expr.p}, {expr.q})"
    
    # Handle floating-point numbers by converting them to rational
    elif isinstance(expr, sp.Float):
        rational = sp.Rational(expr).limit_denominator()
        if rational.q == 1:
            return str(rational.p)
        return f"div({rational.p}, {rational.q})"
    
    # Handle symbols (e.g., variables)
    if isinstance(expr, sp.Symbol):
        return str(expr)
    
    # Handle other types of numbers (e.g., integers, floats)
    for item in numbers_types:
        if isinstance(expr, item):
            return str(expr)
    
    # If expression type is unsupported, raise an error
    print(expr)
    raise ValueError(f"Unsupported expression type: {type(expr)}")
    
def convert_to_sympy_expression(func_form):
    """
    Convert a functional form string back into a SymPy expression.
    
    Parameters:
    - func_form: String in functional form (e.g., 'mul(s_1, s_2, s_4, pow(s_3, -1))')
    
    Returns:
    - SymPy expression.
    """
    def _parse_expr(expression):
        for func, sympy_func in functional_to_sympy.items():
            if expression.startswith(f"{func}(") and expression.endswith(")"):
                # Extract the arguments
                args_str = expression[len(func) + 1:-1]
                args = _split_args(args_str)
                # Recursively parse the arguments
                parsed_args = [_parse_expr(arg) for arg in args]
                return sympy_func(*parsed_args)
        
        # If it's a variable (e.g., s_1), return it as a SymPy symbol
        if expression.startswith('s_'):
            return sp.Symbol(expression)
        
        # If it's a number, parse it as a SymPy number
        try:
            return sp.sympify(expression)
        except sp.SympifyError:
            raise ValueError(f"Unsupported expression format: {expression}")

    def _split_args(args_str):
        """
        Split the argument string into individual arguments, considering nested functions.
        """
        args = []
        current_arg = []
        depth = 0

        for char in args_str:
            if char == ',' and depth == 0:
                args.append(''.join(current_arg).strip())
                current_arg = []
            else:
                current_arg.append(char)
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1

        args.append(''.join(current_arg).strip())
        return args

    return _parse_expr(func_form)
