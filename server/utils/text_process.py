import re

def text_pr(old, new):
    o = old.lower()
    n = new.lower()

    o_clean = re.sub(r"[ ,.]", "", o)
    n_clean = re.sub(r"[ ,.]", "", n)

    i = 0
    while i < len(o_clean) and i < len(n_clean) and o_clean[i] == n_clean[i]:
        i += 1

    return old[:i] + new[i:]