"""GBNF grammar for the DSL, generated from the dataset catalogs.

llama.cpp supports `grammar`; Ollama's API only takes JSON schemas, so grammar-constrained
decoding runs through llama.cpp / llama-cpp-python.
"""

from .formats import BARE, catalog


def _alts(words):
    return " | ".join(f'"{w}"' for w in sorted(words))


def dsl_gbnf(cat=None):
    cat = cat or catalog()
    toppings = sorted(e.lower() for e in cat.ids["TOPPING"])
    styles = sorted(e.lower() for e in cat.ids["STYLE"])
    sizes = sorted(e.lower() for e in cat.ids["SIZE"])
    drinks = sorted(e.lower() for e in cat.ids["DRINKTYPE"])
    containers = sorted(e.lower() for e in cat.ids["CONTAINERTYPE"])
    numbers = sorted(cat.ids["NUMBER"], key=lambda x: (len(x), x))
    volumes = sorted(f"{a.strip()}{u.strip().lower()}" for a, u in cat.volumes)

    return "\n".join(
        [
            'root        ::= suborder (" ; " suborder)*',
            "suborder    ::= pizza | drink",
            'pizza       ::= "p" (" " pizzaattr)*',
            'drink       ::= "d" (" " drinkattr)*',
            "pizzaattr   ::= number | size | topping | style | complex | negated",
            "drinkattr   ::= number | size | drinktype | container | volume",
            'negated     ::= "!" (topping | style | complex)',
            'complex     ::= ("+" | "~") topping',
            f"number      ::= {_alts(numbers)}",
            f"size        ::= {_alts(sizes)}",
            f"topping     ::= {_alts(toppings)}",
            f'style       ::= "^" ({_alts(styles)})',
            f"drinktype   ::= {_alts(drinks)}",
            f"container   ::= {_alts(containers)}",
            f"volume      ::= {_alts(volumes)}",
            "",
        ]
    )


def write_gbnf(path, cat=None):
    text = dsl_gbnf(cat)
    with open(path, "w") as f:
        f.write(text)
    return text
