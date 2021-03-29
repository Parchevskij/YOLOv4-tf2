from functools import wraps, reduce

def compose(*funcs):
  return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)