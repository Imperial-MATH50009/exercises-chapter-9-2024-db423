"""Implement the code."""
from functools import singledispatch  # NOQA F401
import numbers


class Expression:
    """Implement expressions."""

    precedence = float('inf')

    def __init__(self, *operands):
        self.operands = operands
    
    def __add__(self, other):
        if not isinstance(other, Expression):
            other = Number(other)
        return Add(self, other)
    
    def __sub__(self, other):
        if not isinstance(other, Expression):
            other = Number(other)
        return Sub(self, other)
    
    def __mul__(self, other):
        if not isinstance(other, Expression):
            other = Number(other)
        return Mul(self, other)
    
    def __truediv__(self, other):
        if not isinstance(other, Expression):
            other = Number(other)
        return Div(self, other)
    
    def __pow__(self, other):
        if not isinstance(other, Expression):
            other = Number(other)
        return Pow(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rsub__(self, other):
        if not isinstance(other, Expression):
            other = Number(other)
        return Sub(other, self)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __rtruediv__(self, other):
        if not isinstance(other, Expression):
            other = Number(other)
        return Div(other, self)
    
    def __rpow__(self, other):
        if not isinstance(other, Expression):
            other = Number(other)
        return Pow(other, self)

class Operator(Expression):
    """Base class for all operator nodes."""
    symbol = None  # To be defined by subclasses
    precedence = 0  # Default precedence
    
    def __init__(self, *operands):
        if len(operands) < 1:
            raise ValueError("Operators must have at least one operand")
        super().__init__(*operands)
    
    def __str__(self):
        parts = []
        for operand in self.operands:
            # Handle precedence and parentheses
            if isinstance(operand, Operator) and operand.precedence < self.precedence:
                parts.append(f"({operand})")
            else:
                parts.append(str(operand))
        
        return f" {self.symbol} ".join(parts)

class Terminal(Expression):
    """Base class for terminal nodes (values with no operands)."""
    def __init__(self, value):
        super().__init__()
        self.value = value
    
    def __repr__(self):
        return f"{type(self).__name__}({repr(self.value)})"
    
    def __str__(self):
        return str(self.value)
    
    def _to_string(self, parent_precedence=float('inf')):
        return str(self.value)
    
class Number(Terminal):
    """Terminal node representing a numeric value."""
    precedence = float('inf')  # Highest precedence
    
    def __init__(self, value):
        if not isinstance(value, (int, float, complex)):
            raise ValueError("Number value must be numeric")
        super().__init__(value)


class Symbol(Terminal):
    """Terminal node representing a symbolic variable."""
    precedence = float('inf')  # Highest precedence
    
    def __init__(self, value):
        if not isinstance(value, str):
            raise ValueError("Symbol value must be a string")
        super().__init__(value)

class Add(Operator):
    symbol = '+'
    precedence = 1

class Sub(Operator):
    symbol = '-'
    precedence = 1

class Mul(Operator):
    symbol = '*'
    precedence = 2

    def __str__(self):
        # Separate constant and non-constant operands
        constants = []
        others = []
        for operand in self.operands:
            if isinstance(operand, Number):
                constants.append(operand)
            else:
                others.append(operand)
        
        # Format constants first, then other operands
        parts = [str(op) for op in constants + others]
        return " * ".join(parts)
    
class Div(Operator):
    symbol = '/'
    precedence = 2

class Pow(Operator):
    symbol = '^'
    precedence = 3  

def postvisitor(expr, fn, **kwargs):
    '''Visit an Expression in postorder applying a function to every node.
    
    Parameters
    ----------
    expr: Expression
        The expression to be visited.
    fn: function(node, *o, **kwargs)
        A function to be applied at each node. The function should take the node 
        to be visited as its first argument, and the results of visiting its 
        operands as any further positional arguments. Any additional information 
        that the visitor requires can be passed in as keyword arguments.
    **kwargs:
        Any additional keyword arguments to be passed to fn.
        
    Returns
    -------
    The result of applying fn to the expression and its visited operands.
    '''
    visited = {}  # Memoization dictionary to store already visited nodes
    
    stack = [(expr, False)]  # Stack items are tuples of (node, processed)
    
    while stack:
        node, processed = stack.pop()
        
        if node in visited:
            continue
            
        if processed:
            # Get the results for all operands from the visited dict
            operand_results = tuple(visited[c] for c in node.operands)
            # Store the result of applying fn to this node
            visited[node] = fn(node, *operand_results, **kwargs)
        else:
            # Push the node back as processed=True
            stack.append((node, True))
            # Push all children in reverse order (so they're processed left-to-right)
            for child in reversed(node.operands):
                if child not in visited:  # Only push if not already visited
                    stack.append((child, False))
    
    return visited[expr]

# Differentiation Functions
@singledispatch
def differentiate(expr, *, var):
    """Differentiate an expression with respect to a given variable."""
    raise NotImplementedError(
        f"Cannot differentiate a {type(expr)._name_}"
    )


@differentiate.register(Number)
def _(expr, *, var):
    return Number(0)  # Derivative of a constant is 0


@differentiate.register(Symbol)
def _(expr, *, var):
    return Number(1) if expr.value == var else Number(0)
# 1 if variable matches


@differentiate.register(Add)
def _(expr, *operands, var):
    # Differentiate sum: (f + g)' = f' + g'
    return Add(*[differentiate(op, var=var) for op in expr.operands])


@differentiate.register(Mul)
def _(expr, *operands, var):
    # Differentiate product: (f * g)' = f' * g + f * g'
    f, g = expr.operands
    return Add(
        Mul(differentiate(f, var=var), g),  # f' * g
        Mul(differentiate(g, var=var), f)   # f * g'
    )


@differentiate.register(Div)
def _(expr, *operands, var):
    # Differentiate division: (f / g)' = (f' * g - f * g') / g^2
    f, g = expr.operands
    return Div(
        Sub(
            Mul(differentiate(f, var=var), g),  # f' * g
            Mul(differentiate(g, var=var), f)   # f * g'
        ),
        Pow(g, Number(2))  # g^2
    )


@differentiate.register(Pow)
def _(expr, *operands, var):
    """Differentiate f^n: (f^n)' = n * f^(n-1) * f'."""
    base, exponent = expr.operands
    if isinstance(exponent, Number):  # Constant exponent
        return Mul(
            Mul(exponent, Pow(base, Sub(exponent, Number(1)))),  # n * f^(n-1)
            differentiate(base, var=var)  # * f'
        )
    else:
        raise NotImplementedError("Variable exponents are not yet supported.")