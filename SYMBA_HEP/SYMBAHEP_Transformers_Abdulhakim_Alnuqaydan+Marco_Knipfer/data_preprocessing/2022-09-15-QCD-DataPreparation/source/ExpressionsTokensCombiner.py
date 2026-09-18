import re
import sympy as sp
from icecream import ic

def fix_i(expr_str):
    reg_ex = "[^a-z]i[^a-z,^\d]"
    replaced = re.sub(reg_ex, fix_i_match, expr_str)
    return replaced
    
def fix_i_match(matchobj):
    """
    i --> I
    """
    match = matchobj.group(0)
    return match.replace("i", "I")
    

def combine_m_match(matchobj):
    """
    `m_tt ^ x --> mxtt` where x is an integer and tt is top in this example
    makes sequences shorter
    """
    match = matchobj.group(0).split("**")
    exponent = match[1]
    m, m_name = match[0].split("_")
    return m+exponent+m_name
    # else:
    #     return m+"2"+m_name + "**" + str(exponent)

def combine_m_s_match(matchobj):
    """
    `m_tt2**y*s_12 --> m2_ttyxs_12` makes sequences shorter
    """
    x = matchobj.group(0).split("**")
    if len(x) == 2:
        m = x[0]
        exp_and_s = x[1]
        exp, s = exp_and_s.split("*")
        return m+exp+"x"+s
    else:
        m, s = x[0].split("*")
    return m+"x"+s

def combine_m(expr_str):
    reg_ex = "(m\_[a-z]{1,})\*\*(\d{1,})"
    replaced = re.sub(reg_ex, combine_m_match, expr_str)
    return replaced

def combine_m_s(expr_str):
    reg_ex = "m\d[a-z]{1,}(\*\*\d{1,})?\*s\_\d\d"
    replaced = re.sub(reg_ex, combine_m_s_match, expr_str)
    return replaced

def shorten_expression(expr):
    """
    Sympy expression -> strings, shorten --> sympy expression
    """
    expr_str = str(expr)
    expr_str = fix_i(expr_str)
    expr_str = combine_m(expr_str)
    expr_str = combine_m_s(expr_str)
    return sp.sympify(expr_str)
