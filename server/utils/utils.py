DEBUG = True
def dprint(*a, **k): 
    if DEBUG: print(*a, **k)

LOGG = True
def lprint(*a, **k): 
    if LOGG: print(*a, **k)