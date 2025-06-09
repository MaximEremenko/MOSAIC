# -*- coding: utf-8 -*-
"""
Created on Thu Apr 24 15:09:37 2025

@author: Maksim Eremenko
"""

# utilities/logic_parser.py

import re
import sympy as sp

# 1) Symbols for h, k, l
h, k, l = sp.symbols('h k l')
symbol_map = {'h': h, 'k': k, 'l': l}

# 2) The locals dict passed into sympify / lambdify
allowed_locals = {
    'pi':    sp.pi,
    'E':     sp.E,
    'I':     sp.I,
    'oo':    sp.oo,
    'infty': sp.oo,
    'sin':   sp.sin,
    'cos':   sp.cos,
    'tan':   sp.tan,
    'asin':  sp.asin,
    'acos':  sp.acos,
    'atan':  sp.atan,
    'atan2': sp.atan2,
    'sinh':  sp.sinh,
    'cosh':  sp.cosh,
    'tanh':  sp.tanh,
    'asinh': sp.asinh,
    'acosh': sp.acosh,
    'atanh': sp.atanh,
    'exp':   sp.exp,
    'log':   sp.log,
    'ln':    sp.log,
    'sqrt':  sp.sqrt,
    'abs':   sp.Abs,
    'Abs':   sp.Abs,
    'floor': sp.floor,
    'ceiling': sp.ceiling,
    'min':   min,
    'max':   max,
    'Rational':  sp.Rational,
    'factorial': sp.factorial,
    'gamma':     sp.gamma,
    'erf':       sp.erf,
    'erfc':      sp.erfc,
    'Ei':        sp.Ei,
}

def preprocess(expr: str) -> str:
    """
    Clean up a user‐supplied string:
      - replace π with 'pi'
      - insert '*' between 'pi' and a following variable
      - convert '^n' to '**n'
    """
    expr = expr.replace("π", "pi")
    # catch cases like 'pi h' → 'pi*h'
    expr = re.sub(r"pi([A-Za-z])", r"pi*\1", expr)
    # exponentiation caret → Python power
    expr = re.sub(r"(\b[A-Za-z]\w*)\^(\d+)", r"\1**\2", expr)
    return expr

def parse_logic(s: str, symbols: dict, locals_dict: dict):
    """
    Recursively parse a logical expression involving 'and', 'or', 'xor'
    and relational/arithmetic atoms into a single Sympy Boolean expression.
    """
    s = s.strip()

    def split_top(expr, sep):
        parts, depth, cur = [], 0, ""
        i = 0
        while i < len(expr):
            c = expr[i]
            if c == "(":
                depth += 1; cur += c; i += 1
            elif c == ")":
                depth -= 1; cur += c; i += 1
            elif depth == 0 and expr.startswith(sep, i):
                parts.append(cur); cur = ""; i += len(sep)
            else:
                cur += c; i += 1
        parts.append(cur)
        return parts

    def atom(tok):
        tok = tok.strip()
        # strip outer parentheses if they balance
        if tok.startswith("(") and tok.endswith(")"):
            depth = 0
            for idx,ch in enumerate(tok):
                if ch=="(": depth+=1
                elif ch==")": depth-=1
                if depth==0 and idx < len(tok)-1:
                    break
            else:
                return parse_or(tok[1:-1])
        # relational or arithmetic → sympify
        if re.search(r"[<>=\+\-\*/]", tok):
            return sp.sympify(tok, locals=locals_dict)
        # single symbol
        if tok in symbols:
            return symbols[tok]
        raise ValueError(f"Unknown token: {tok}")

    def parse_and(expr):
        parts = split_top(expr, " and ")
        if len(parts) > 1:
            return sp.And(*[parse_or(p) for p in parts])
        return atom(expr)

    def parse_xor(expr):
        parts = split_top(expr, " xor ")
        if len(parts) > 1:
            return sp.Xor(*[parse_and(p) for p in parts])
        return parse_and(expr)

    def parse_or(expr):
        parts = split_top(expr, " or ")
        if len(parts) > 1:
            return sp.Or(*[parse_xor(p) for p in parts])
        return parse_xor(expr)

    return parse_or(s)
