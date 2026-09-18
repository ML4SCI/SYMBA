from nltk.tree import Tree
import sympy as sp
from icecream import ic
import time
import copy

sqampls_file = "../../data.nosync/QED_amplitudes_TreeLevel_2to2.txt"
ampls_raw_file = "../../data.nosync/QED_amplitudes_TreeLevel_2to2_raw.txt"


# the amplitudes are already in hybrid prefix notation. Now writing a script to convert them to a tree is kind
# of stupid, because I had them as trees before I converted them.
# Sadly there are so many steps in between and I don't want to change them now.
# GOAL: Convert to
# - tree
# - prefix
# Since I think it easier to first go to the tree and then prefix, I'll do the tree here.


operators = { "ee": 3,
        "ee^(*)": 3,
        "b": 3,
        "b^(*)": 3,
        "A": 3,
        "A^(*)": 3,
        "Prod": 2,
        "Prod(": -1,  # )
        "Pow": 2,
        "gamma": 3,
        "Sum": 2,
        "Sum(": -1,  # )
}


# basis functions and number of indices
basis_functions = {
        "e": 2,
        "mu": 2,
        "t": 2,
        "u": 2,
        "d": 2,
        "s": 2,
        "c": 2,
        "tt": 2,
        "b": 2,
        "A": 2,
        }


indices_roman = [
        "i",
        "j",
        "k",
        "l",
        ]


indices_greek = [
        "alpha",
        "beta",
        "gam",
        "del",
        "eta",
        "sigma",
        "tau",
        "rho",
        "lambda",
        "nu",
        "mu",
        "eps",
        ]


def raw_ampl_to_tree(raw_ampl):
    raw_ampl_split = raw_ampl.split(";")
    tree_raw = get_tree(raw_ampl_split)
    tree = ampl_raw_tree_to_nltk(tree_raw)
    tree = nltk_tree_expand_subscripts(tree)
    return tree


def ampl_to_tree(arr, remove_hybrid_parentheses=False):
    """convert hybrid prefix amplitude to nltk.Tree
    Hybrid prefix is either normal prefix or prefix where there are products/sums with
    arbitrary number of arguments. They work like this:
        ["Prod(", "a1", "a2", "a3", ")"]
    Watch out, if you `remove_hybrid_parentheses` and then do the round-trip with
    `tree_to_prefix` with `hybrid=False`, then the result is wrong, because
    there will be things like [Prod(, 1, 2, 3, )] --> [Prod, 1, 2, 3]
    """

    if len(arr) == 1:
        return arr[0]

    op_pos = rightmost_operator_pos(arr)
    if (op_pos == -1) | (op_pos == len(arr)):
        print("something went wrong, operator should not be at end of array")

    op = arr[op_pos]
    number_of_args = operators[op]
    if number_of_args == -1:
        bracket_pos = next_bracket_pos(arr, op_pos)
        func_and_args = arr[op_pos:bracket_pos]
        if remove_hybrid_parentheses:
            func_and_args[0] = func_and_args[0][:-1]
        foo = func_to_tree(func_and_args)
        arr_new = arr[:op_pos] + [foo] + arr[bracket_pos+1:]
    else:
        foo = func_to_tree(arr[op_pos:op_pos+number_of_args+1])
        arr_new = arr[:op_pos] + [foo] + arr[op_pos+number_of_args+1:]
    return ampl_to_tree(arr_new, remove_hybrid_parentheses=remove_hybrid_parentheses)


def next_bracket_pos(expr_arr, op_pos):
    pos = op_pos + 1
    x = expr_arr[pos] 
    while x != ")":
        pos = pos + 1
        x = expr_arr[pos]
        if pos > len(expr_arr):
            print("Error in next_bracket_pos")
            return -1
    return pos


def rightmost_operator_pos(expr, pos=-1):
    """
    from expression get the rightmost operator position.
    The operators are in the global dictionary `operators`.

    Notes:
        - if `expr[pos]` is already a Tree, then checking if it's in `operators.keys()` fails,
        thus first check if it's a tree. (Tree not hashable)
    """
    if isinstance(expr[pos], Tree):
        return rightmost_operator_pos(expr, pos-1)
    if expr[pos] in operators.keys():
        return len(expr) + pos
    else:
        return rightmost_operator_pos(expr, pos-1)


def func_to_tree(arr):
    func = arr[0]
    args = arr[1:]
    return Tree(func, args)


def tree_to_prefix(tree_input, hybrid=False):
    """converts a tree to an array in prefix notation.
    Automatically detects hybrid prefix if an operator is like `Prod(` )
    If operators don't have parentheses, you can still go to hybrid prefix with
    `hybrid=True`. Default is `False`
    """
    tree = copy.deepcopy(tree_input)
    if not hybrid:
        tree = expand_tree(tree)
    if hybrid:
        tree = contract_tree(tree)
    arr = []
    node = tree.label()
    if hybrid and (node in ["Sum", "Prod"]) and (len(tree)>2):
        node = node + "("  # )
    arr.append(node)
    for i in range(len(tree)):
        if isinstance(tree[i], Tree):
            arr = arr + tree_to_prefix(tree[i], hybrid=hybrid)
        else:
            arr.append(tree[i])

    if node[-1] == "(":  # )
        arr.append(")")
    return arr


def expand_tree(tree_input):
    """
    If the tree has Prod or Sum nodes with more than 2 arguments
    --> expand them to only have 2
    """
    tree = copy.deepcopy(tree_input)  # is there a better way to not mutate tree?
    if not isinstance(tree, Tree):
        return tree
    node = tree.label()
    nodes_considered = ["Sum", "Sum(", "Prod", "Prod("]  # ))
    if node[-1] == "(":  # )
        node = node[:-1]
        tree.set_label(node)
    if (node in nodes_considered) and (len(tree) > 2):
        leaves = [expand_tree(t) for t in tree[1:]]
        tree[0] = expand_tree(tree[0])
        tree[1] = expand_tree(Tree(node, leaves))
        del tree[2:]
    else:
        tree[0:] = [expand_tree(t) for t in tree[0:]]
    return tree


def contract_tree(tree_input, runs=0, add_opening_bracket=True):
    """inverse of `expand_tree`, not fully tested.

    Note: This implementation is not fully tested and probably not working 100%.
    One problem was that it didn't work at all depths.
    Right now my workaround is that I just run it three times and hope all
    cases are caught, but I don't think so.
    """

    tree = copy.deepcopy(tree_input)  # don't mutate tree

    if runs>2:
        return tree
    if not isinstance(tree, Tree):
        return tree
    node = tree.label()
    nodes_considered = ["Sum", "Sum(", "Prod", "Prod("]  # ))
    if (node in nodes_considered) and (isinstance(tree[1], Tree)) and (tree[1].label() == node):
        if add_opening_bracket:
            node = node + "("  # )
        tree.set_label(node)
        subtree = contract_tree(tree[1], add_opening_bracket=add_opening_bracket)
        # del[tree[1:]]
        tree[0:] = tree[0:1] + subtree[0:]
    else:
        tree[0:] = [contract_tree(t, add_opening_bracket=add_opening_bracket) for t in tree[0:]]
    return contract_tree(tree, runs=runs+1, add_opening_bracket=add_opening_bracket)


def get_tree(expression):
    last_open_bracket_idx = get_last_open_bracket(expression)
    while last_open_bracket_idx != -1:
        next_closing_bracket_idx = get_next_closing_bracket(expression, last_open_bracket_idx)
        sub_expr = expression[last_open_bracket_idx+1:next_closing_bracket_idx]
        sub_expr = [expression[last_open_bracket_idx-1]] + sub_expr  # add operator before ()
        expression[last_open_bracket_idx-1] = sub_expr
        del expression[last_open_bracket_idx:next_closing_bracket_idx+1]
        last_open_bracket_idx = get_last_open_bracket(expression)

    return expression[0]


def get_tree_old(expression):
    last_open_bracket_idx = get_last_open_bracket(expression)
    ic(last_open_bracket_idx)
    while last_open_bracket_idx != -1:
        next_closing_bracket_idx = get_next_closing_bracket(expression, last_open_bracket_idx)
        sub_expr = expression[last_open_bracket_idx+1:next_closing_bracket_idx]
        sub_expr = [expression[last_open_bracket_idx-1]] + sub_expr  # add operator before ()
        expression[last_open_bracket_idx-1] = sub_expr
        del expression[last_open_bracket_idx:next_closing_bracket_idx+1]
        last_open_bracket_idx = get_last_open_bracket(expression)
        ic(last_open_bracket_idx)

    return expression[0]


def get_last_open_bracket(expression):
    for i in range(len(expression)):
        if expression[-i] == '(':
            return len(expression) - i
        else:
            pass
    return -1


def get_next_closing_bracket(expression, last_open_bracket_idx):
    for i in range(last_open_bracket_idx+1, len(expression)):
        if expression[i] == ")":
            return i
    return -1

def ampl_raw_tree_to_nltk(lst):
    """convert raw amplitude tree (nested list) to nltk tree
    Example:
        In: ['Prod',
           '-1/2',
           'i',
           ['Pow', 'e', '2'],
           ['Pow',
            ['Sum',
             ['Pow', 'm_e', '2'],
             ['Prod', '-1', 's_13'],
             ['Prod', '1/2', 'reg_prop']],
            '-1'],
           'gamma_{+%\\sigma_126,%eps_36,%del_171}',
           'gamma_{%\\sigma_126,%eta_132,%del_172}',
           'e_{i_3,%del_171}(p_1)_u',
           'e_{k_3,%del_172}(p_2)_u',
           'e_{l_3,%eps_36}(p_3)_u^(*)',
           'e_{i_5,%eta_132}(p_4)_u^(*)']

        Out: 
            Tree('Prod', ['-1/2', 'i', Tree('Pow', ['e', '2']), Tree('Pow', [Tree('Sum', [Tree('Pow', ['m_e', '2']), Tree('Prod', ['-1', 's_13']), Tree('Prod', ['1/2', 'reg_prop'])]), '-1']), 'gamma_{+%\\sigma_126,%eps_36,%del_171}', 'gamma_{%\\sigma_126,%eta_132,%del_172}', 'e_{i_3,%del_171}(p_1)_u', 'e_{k_3,%del_172}(p_2)_u', 'e_{l_3,%eps_36}(p_3)_u^(*)', 'e_{i_5,%eta_132}(p_4)_u^(*)'])
    """
    op = lst[0]
    args = lst[1:]
    args = [ampl_raw_tree_to_nltk(a) if isinstance(a, list) else a for a in args]
    tree = Tree(op, args)
    return tree


def nltk_tree_expand_subscripts(tree):
    """
    go through each leaf in tree, check if has subscript, if yes, expand in subtree with subscripts as leaves
    """
    if not isinstance(tree, Tree):
        if has_subscript(tree):
            return subscripts_to_subtree(tree)
        return tree
    label = tree.label()
    leaves = tree[0:]
    leaves_expanded = [nltk_tree_expand_subscripts(l) for l in leaves]
    return Tree(label, leaves_expanded)


def has_subscript(str):
    """
    ic(has_subscript("asdf")) --> False
    ic(has_subscript("asdf_{a, b, c}")) --> True
    ic(has_subscript("p_mu")) --> True
    """
    ret = ("_" in str) and ("{" in str)  # }
    ret = (ret or ("p_" in str))
    return ret



def subscripts_to_subtree(expr, save_input=False):
    """
    Turn expression like `gamma_{a, b, c}` to a tree
    A `%` indicates that it's an index
    """
    expr_new = remove_unnecessary_in_indices(expr)
    var, subscript = expr_new.split("_", maxsplit=1)
    if is_basis_func(expr):
        return basis_function_to_subtree(expr)

    if var == "gamma":
        subscripts = format_gamma(subscript)
        return Tree(var, subscripts)

    if var == "p":
        ret = p_sub_to_tree(expr_new)
        return ret
    

    # if you're sure that your input is correct in the form 'asdf_{a, b, c}',
    # then return `Tree[asdf, [a,b,c]]`
    if save_input:
        indices = subscript[1:-1].split(",")
        indices = ["%" + i for i in indices]
        return Tree(var, indices)

    # if non of the above, just split at first `_`. The thing before is the node and
    # there is only one subscript, namely everything after the `_`.
    return Tree(var, [subscript])


def p_sub_to_tree(p_expr):
    """
    from something like p_4_sigma_241 make Tree(p_4, [%sigma_241])
    """
    p_expr = remove_unnecessary_in_indices(p_expr)
    if len(p_expr.split("_")) == 3:
        p, num, index = p_expr.split("_")
        label = p + "_" + num
        index = "%" + index
        return Tree(label, [index])
    elif len(p_expr.split("_")) == 4:
        p, num, index, index_num = p_expr.split("_")
        label = p + "_" + num
        index = "%" + index + "_" + index_num
        return Tree(label, [index])

def is_basis_func(str):
    """
    check if given string is in the form of a basis function, i.e.
        e_{a1, a2, a3}(p_1)_u
    """

    s = str.split("_", maxsplit=1)
    if not s[0] in basis_functions.keys():
        return False
    if not s[1][0] == "{":  #}
        return False
    s[1] = s[1][1:]
    s[1] = s[1].split("}")
    if s[1][1] == '':
        return False
    indices = s[1][0].split(",")
    basis_func = s[0]
    num_indices = basis_functions[basis_func]
    if len(indices) != num_indices:
        return False
    return True


def basis_function_to_subtree(str):
    """assumes a valid basis function in the form
        'e_{l_3,%eps_36}(p_3)_u^(*)'
    Returns a nltk.Tree with the indices and the momentum as leaves and 
    e_u^* as node.
    
    - "^(*)": this goes to the end of the node label
    - indices: will all become leaves
    """
    basis_f, rest = str.split("}", maxsplit=1)
    basis_f = basis_f + "}"   # `e_{i, alpha}`
    basis_f_tree = subscripts_to_subtree(basis_f, save_input=True)

    # momentum (p_1)_u or (p_1)_u^(*)
    momentum, particle = rest.split(")", maxsplit=1)
    momentum = momentum + ")"

    # ^(*)
    if len(particle.split("^")) == 2:
        particle_name, star = particle.split("^")
        label = basis_f_tree.label()
        basis_f_tree.set_label(label + particle_name)
        label = basis_f_tree.label()
        basis_f_tree.set_label(label + "^" + star)
    elif len(particle.split("^")) == 1:
        label = basis_f_tree.label()
        basis_f_tree.set_label(label + particle)

    basis_f_tree[0:] = basis_f_tree[0:] + [momentum]

    return basis_f_tree


def format_gamma(subscript):
    """
    input looks like: '{%\\sigma_49,%gam_44,%eta_12}'
    gamma alsways has 3 indices
    """
    ind1, ind2, ind3 = subscript[1:-1].split(",")
    ind1 = "%" + ind1
    ind2 = "%" + ind2
    ind3 = "%" + ind3

    return [ind1, ind2, ind3]


def remove_unnecessary_in_indices(expr):
    expr_new = expr.replace("\\", "")
    expr_new = expr_new.replace("%", "")
    expr_new = expr_new.replace("+", "")
    return expr_new


def rename_indices(tree):
    """
    For a finished nltk.Tree, make indices easier.
    By easier I mean:
    The indices are in the form `{'%eta_132', '%del_172', '%sigma_126'}`,
    but the numbers are too big and this is not handy for a neural network,
    because there will be too many different tokens.
    The indices are all collected and then categorized by what is in front of the `_`.
    Then the number behind the `_` are enumerated and changed such that they start from 0.
    
    Of course the expression should have only dummy indices, otherwise this will probably break something.
    Could implement a check if the index appears at least (or exaclty?) twice.
    """
    indices = collect_indices(tree)
    index_categorization = categorize_indices(indices)
    index_replacements = get_index_replacements(index_categorization)

    tree = nltk_tree_replace_leaves(tree, index_replacements)

    return tree


def nltk_tree_replace_leaves(tree: Tree, index_replacements: dict):
    """
    For a tree, go threough each leaf and replace it if the string is in index_replacements
    """
    for leafPos in tree.treepositions('leaves'):
        leaf = tree[leafPos]
        if leaf in index_replacements.keys():
            tree[leafPos] = index_replacements[leaf]
    return tree

def get_index_replacements(index_categorization: dict):
    """
    For a given index_categorization (from categrorize_indices),
    get the replacements how to rename the indices to have "low" numbers 
    E.g. this
        {
        '%del': ['171', '172'],
         '%eps': ['36'],
         '%eta': ['132'],
         '%i': ['3', '5'],
         '%k': ['3'],
         '%l': ['3'],
         '%sigma': ['126']
        }
    turns into this
        {
        '%del_171': `del_0`,
        '%del_172': `del_1`,
        `%eta_132`: `eta_0`,
        ...
        }
    """
    index_replacements = dict()
    for key, values in index_categorization.items():
        for i, i_old in enumerate(values):
            if i_old == "MISSING":  # basically means index had no `_`
                index_replacements[key] = key + "_" + str(i)
            else:
                index_replacements[key+"_"+str(i_old)] = key + "_" + str(i)
    return index_replacements
            

def categorize_indices(indices: set):
    """
    The indices come in a set like this:
        {'%del_171', '%del_172', '%eps_36', '%eta_132', '%i_3', '%i_5', '%k_3', '%l_3', '%sigma_126'}
    Return a dictionary where they are all categorized by the string in front of the `_`.
    Above set gives
        {
        '%del': ['171', '172'],
         '%eps': ['36'],
         '%eta': ['132'],
         '%i': ['3', '5'],
         '%k': ['3'],
         '%l': ['3'],
         '%sigma': ['126']
        }

    If an index appears without `_`, e.g. `%a` instead of `%a_1`,
    then it is treated as if it was `a_MISSING` with the string "MISSING" as index.
    If you are using MISSING as an index, well ...
    """
    categorization = dict()
    for index in indices:
        index_split = index.split("_", maxsplit=1)
        if (len(index_split) == 1):  # no _ in index
            if not index_split[0] in categorization.keys():
                categorization[index_split[0]] = ["MISSING"]
            else:
                categorization[index_split[0]].append("MISSING")
        else:
            name, number = index_split
            if not name in categorization.keys():
                categorization[name] = [number]
            else:
                categorization[name].append(number)

    for key in categorization.keys():
        categorization[key].sort()
    return categorization



def collect_indices(tree, indices=None):
    """collect all indices in a nltk.Tree
    Indices are marked by having a `%`.

    Feels kind of stupid writing my own search for this, but
    I could not find it anywhere.
    """
    if indices is None:
        indices = set()

    if isinstance(tree, str):
        if is_index(tree):
            indices.add(tree)
            return indices

    if isinstance(tree, Tree):
        for leaf in tree.leaves():
            # indices.add(collect_indices(leaf))
            indices = collect_indices(leaf, indices=indices)

    return indices

def is_index(s: str):
    """
    Something is an index if it starts with `%`.
    """
    if not isinstance(s, str):
        return False
    if s == "":
        return False
    if not s[0] == "%":
        return False
    if len(s.split("%")) > 2:
        return False

    return True


if __name__ == "__main__":
    with open(ampls_raw_file) as f:
        ampls_raw = f.readlines(100000)
        ampls_raw = [a[:-1] for a in ampls_raw]
        
    exp = ampls_raw[0]
    exp = exp.split(";")
    print(exp)

    tree_raw = get_tree(exp)
    ic(tree_raw)
    tree = ampl_raw_tree_to_nltk(tree_raw)
    tree.pretty_print(abbreviate=True, unicodelines=True)
    # tree = fix_tree(tree)
    # ic(tree)
    # final_expr = fix_subscripts(tree)
    # ic(final_expr)

