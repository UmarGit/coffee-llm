"""Negative training data for PIZZA: teach the model that only the menu exists.

PIZZA has no way to say "this is not an order", so a model trained on it always
prints a ticket: "hehe hehe" becomes a pizza in the "hehe hehe" style. The café
task fixed the same problem by teaching a reject symbol on a generated, noisy,
negative-heavy mix; this does the same for PIZZA, deliberately strict.

The rule is deliberately strict:

    an order made only of menu items   ->  its ticket   (unchanged PIZZA)
    anything else at all               ->  ?

including an order with anything extra in it: "a large ham pizza and a dog" and
"a hehe pizza" are refused whole, not half-parsed. The consumer never has to
decide which half of a ticket to trust.

Kinds of negative, all generated rather than listed, so the model learns the rule
rather than a phrasebook:

    non       chatter and questions        "where is the restroom"           ?
    off       off-menu requests            "two cats and a dog"              ?
    junk      gibberish, any script        "hehe hehe", "asdkjh", "1 hello world cola"   ?
    long      paragraphs, lorem ipsum      "lorem ipsum dolor sit amet ..."  ?
    mixed     order + off-menu clause      "<order> and some fries"          ?
    unknown   order with an unknown word   "a large hehe pizza with ham"     ?

Every generated phrase is checked against the PIZZA catalogs: nothing the dataset
can order is ever taught as a refusal. Vocabulary is split, so the held-out
evaluation (`eval_rows`) uses words the model never trained on. `PROBES` is a
fixed challenge set, reported on every run.

Only the ";"-separated formats (words, dsl, tagged) are supported, as in the café
parser; the bracketed formats are refused rather than given an invented encoding.
Dev and test stay pure PIZZA, so every paper number keeps its meaning.
"""

import random
import re
from functools import lru_cache
from pathlib import Path

from .formats import PIZZA_DIR

REJECT = "?"
ARMS = ("words", "dsl", "tagged")

# Share of each kind among the negative rows.
KINDS = {"non": 0.12, "off": 0.15, "junk": 0.25, "long": 0.08, "mixed": 0.20, "unknown": 0.20}

# ---------- vocabulary ----------

GREETINGS = ["hello", "hi", "hey", "hi there", "good morning", "good evening", "yo", "hello hello"]
QUESTIONS = [
    "where is the restroom", "what's the wifi password", "can i pay by card", "do you take cash",
    "what time do you close", "are you open on sunday", "do you deliver", "how long is the wait",
    "what do you recommend", "what's good here", "do you have a menu", "is this gluten free",
    "can i speak to the manager", "where do i pick up", "can i get a receipt", "is parking free",
    "what's your name", "are you a robot", "tell me a joke", "what's the weather like",
    "how was your day", "who won the game", "what time is it", "can you help me",
    "where is the nearest bank", "how do i get downtown", "do you sell gift cards",
    "is anyone there", "can you hear me", "what is this place",
]
CHATTER = [
    "thank you", "thanks so much", "have a nice day", "bye", "goodbye", "see you later",
    "i'm still deciding", "give me a minute", "never mind", "nothing for me", "i already ordered",
    "where is my order", "cancel that", "the usual", "surprise me", "i'm just looking",
    "can you repeat that", "test test", "testing one two three", "yes", "no", "okay", "maybe",
    "i don't know", "sorry wrong number", "i love you", "this is a test", "nothing", "whatever",
]
OFF_ITEMS = [
    "apples", "oranges", "bananas", "grapes", "strawberries", "a fruit salad", "a burger",
    "french fries", "a hot dog", "tacos", "a burrito", "sushi", "ramen", "fried rice",
    "a salad", "a sandwich", "a sub", "pasta", "spaghetti", "lasagna", "soup", "nuggets",
    "a donut", "a muffin", "a croissant", "a bagel", "pancakes", "waffles", "ice cream",
    "a milkshake", "a cake", "cookies", "a brownie", "cheesecake", "a beer", "wine",
    "whiskey", "orange juice", "apple juice", "lemonade", "a smoothie", "hot chocolate",
    "a latte", "a cappuccino", "an espresso", "green tea", "milk", "kombucha", "cola",
    "a phone charger", "a newspaper", "cigarettes", "a gift card", "a taxi", "a umbrella",
    "cats", "a cat", "dogs", "a dog", "a horse", "fish", "a goldfish", "a parrot", "rabbits",
    "a car", "a bicycle", "a laptop", "shoes", "a hat", "books", "a pen", "flowers",
    "a table", "chairs", "a lamp", "batteries", "toothpaste", "socks", "a guitar",
]
NOUNS = [
    "world", "cat", "dog", "house", "tree", "river", "moon", "sun", "car", "phone",
    "computer", "table", "window", "music", "dream", "rocket", "banana", "elephant",
    "monkey", "dragon", "unicorn", "robot", "ocean", "mountain", "cloud", "star",
    "horse", "giraffe", "penguin", "spaceship", "volcano", "zebra",
    "pencil", "book", "chair", "garden", "bridge", "castle", "train", "airplane",
    "planet", "galaxy", "keyboard", "mouse", "pillow", "blanket", "candle", "mirror",
]
ADJECTIVES = [
    "blue", "happy", "sad", "fast", "slow", "loud", "quiet", "shiny", "fuzzy", "purple",
    "angry", "sleepy", "magic", "invisible", "golden", "wooden", "plastic", "frozen",
    "electric", "tiny", "giant", "funny", "weird", "random", "broken", "lucky",
]
LOREM = (
    "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor "
    "incididunt ut labore et dolore magna aliqua ut enim ad minim veniam quis nostrud "
    "exercitation ullamco laboris nisi aliquip ex ea commodo consequat duis aute irure "
    "in reprehenderit voluptate velit esse cillum fugiat nulla pariatur excepteur sint "
    "occaecat cupidatat non proident sunt culpa qui officia deserunt mollit anim id est "
    "laborum quisque faucibus sapien vitae pellentesque placerat cursus pretium tellus"
).split()
FOREIGN = [
    "привет как дела", "где туалет", "спасибо большое", "хочу пиццу пожалуйста",
    "bonjour comment ça va", "merci beaucoup", "hola qué tal", "dónde está el baño",
    "guten tag", "danke schön", "ciao come stai", "konnichiwa", "ni hao", "salaam",
    "obrigado", "dzień dobry", "hej hej", "merhaba",
]
EMOJI = ["🍕", "😂", "🔥", "👍", "❤️", "🙂", "🤔", "🎉", "😎", "🍔"]
SYLLABLES = ["he", "ha", "lo", "la", "bo", "zu", "ki", "ma", "pi", "ra", "to", "nu", "blo", "zor", "gar", "fli"]
KEYS = "qwertyuiopasdfghjklzxcvbnm"
FILLERS = ["", "", "", "", "um ", "uh ", "so ", "ok ", "hey ", "excuse me ", "yeah "]
CLOSERS = ["", "", "", " please", " thanks", " thank you", " now", " lol"]
VERBS = [
    "i want {x}", "can i get {x}", "could i have {x}", "i'd like {x}", "give me {x}",
    "let me get {x}", "i'll have {x}", "{x}", "i need {x}", "do you have {x}",
    "one order of {x}", "two {x}", "three {x}", "{x} and {y}", "{x} with {y}",
]
JOIN = [" and ", " and also ", " plus ", " with a side of ", " and some ", " also "]
NUMBER_WORDS = ["one", "two", "three", "four", "five", "a", "some", "1", "2", "3", "5", "10"]

# Words that name an order, counted as on the menu so junk never contains them.
ORDER_WORDS = {"pizza", "pizzas", "pie", "pies", "drink", "drinks"}
# Catalog forms that are also everyday words ("can i pay"), not menu items here.
AMBIGUOUS = {"can", "cans"}


def _split(items):
    """Every fourth entry is held out of training and used only for evaluation."""
    return [x for i, x in enumerate(items) if i % 4 != 3], [x for i, x in enumerate(items) if i % 4 == 3]


VOCAB = {
    name: _split(items)
    for name, items in {
        "greet": GREETINGS, "question": QUESTIONS, "chatter": CHATTER, "off": OFF_ITEMS,
        "noun": NOUNS, "adj": ADJECTIVES, "foreign": FOREIGN, "syll": SYLLABLES,
    }.items()
}

# ---------- the menu, from the dataset's own catalogs ----------

# Catalogs naming things a customer can order. number.txt and quant_qualifier.txt
# are left out: they map words like "a" and "extra", which every phrase contains.
ENTITY_CATALOGS = ("topping.txt", "style.txt", "drinks.txt", "container.txt",
                   "size.txt", "drink_volume.txt")


@lru_cache(maxsize=None)
def _menu_pattern(pizza_dir=None):
    root = Path(pizza_dir or PIZZA_DIR) / "utils/catalogs"
    forms = set()
    for f in (root / name for name in ENTITY_CATALOGS):
        for line in f.read_text().splitlines():
            phrase = line.split("\t")[0].strip().lower()
            if phrase:
                forms.add(phrase)
    forms = (forms | ORDER_WORDS) - AMBIGUOUS
    alts = "|".join(re.escape(f) for f in sorted(forms, key=len, reverse=True))
    return re.compile(rf"(?<![\w]){'(?:' + alts + ')'}(?![\w])")


def on_menu(text, pizza_dir=None):
    """The first menu phrase (or order word) inside `text`, or None."""
    m = _menu_pattern(pizza_dir).search(text.lower())
    return m.group(0) if m else None


def collisions(pizza_dir=None):
    """Vocabulary entries that name something PIZZA can order. Must be empty."""
    out = []
    for name, (train, held) in VOCAB.items():
        for item in train + held:
            hit = on_menu(item, pizza_dir)
            if hit:
                out.append((name, item, hit))
    return out


# ---------- generators ----------


def _pool(rng, name, held):
    return rng.choice(VOCAB[name][1 if held else 0])


def _nonsense_word(rng, held):
    return "".join(_pool(rng, "syll", held) for _ in range(rng.randint(1, 3)))


def _junk(rng, held):
    r = rng.random()
    if r < 0.18:  # repeated syllables: "hehe hehe", "lolo lolo lolo"
        w = _pool(rng, "syll", held) * rng.randint(2, 3)
        return " ".join([w] * rng.randint(1, 4))
    if r < 0.32:  # keyboard mash
        return " ".join("".join(rng.choice(KEYS) for _ in range(rng.randint(3, 9)))
                        for _ in range(rng.randint(1, 3)))
    if r < 0.46:  # nonsense words
        return " ".join(_nonsense_word(rng, held) for _ in range(rng.randint(1, 5)))
    if r < 0.62:  # word salad, numbers sprinkled in: "1 hello world cola"
        words = [_pool(rng, rng.choice(["noun", "adj"]), held) for _ in range(rng.randint(2, 5))]
        if rng.random() < 0.5:
            words.insert(rng.randrange(len(words) + 1), rng.choice(NUMBER_WORDS))
        if rng.random() < 0.3:
            words.insert(0, _pool(rng, "greet", held))
        return " ".join(words)
    if r < 0.74:  # other languages and scripts
        return _pool(rng, "foreign", held)
    if r < 0.84:  # emoji, alone or with words
        e = "".join(rng.choice(EMOJI) for _ in range(rng.randint(1, 4)))
        return e if rng.random() < 0.5 else f"{e} {_nonsense_word(rng, held)}"
    if r < 0.92:  # digits and symbols
        return " ".join(str(rng.randint(0, 999)) for _ in range(rng.randint(1, 4)))
    return " ".join(rng.sample(LOREM, rng.randint(2, 6)))


def _long(rng, held):
    """A paragraph: lorem ipsum, or a run of chatter and word salad."""
    if rng.random() < 0.5:
        words = [rng.choice(LOREM) for _ in range(rng.randint(15, 60))]
        return " ".join(words)
    parts = []
    for _ in range(rng.randint(3, 7)):
        parts.append(rng.choice([
            _pool(rng, "question", held), _pool(rng, "chatter", held), _junk(rng, held),
            f"the {_pool(rng, 'adj', held)} {_pool(rng, 'noun', held)}",
        ]))
    return " ".join(parts)


def _off_phrase(rng, held):
    x, y = _pool(rng, "off", held), _pool(rng, "off", held)
    return rng.choice(VERBS).format(x=x, y=y)


def _whole(kind, rng, held):
    if kind == "non":
        text = rng.choice([
            lambda: _pool(rng, "greet", held),
            lambda: _pool(rng, "question", held),
            lambda: _pool(rng, "chatter", held),
            lambda: f"{_pool(rng, 'greet', held)} {_pool(rng, 'question', held)}",
        ])()
    elif kind == "off":
        text = _off_phrase(rng, held)
    elif kind == "junk":
        text = _junk(rng, held)
    else:
        text = _long(rng, held)
    if kind in ("non", "off"):
        text = rng.choice(FILLERS) + text + rng.choice(CLOSERS)
    return text.strip()


_HEAD = re.compile(r"\b(pizzas?|pies?)\b")


@lru_cache(maxsize=None)
def pizza_vocab():
    """Every word real PIZZA orders use. Small (a few hundred words): the
    utterances are generated from a closed grammar. A word inserted into an order
    to make it a negative must come from outside this set, or the model would be
    taught to refuse a word that real orders contain."""
    from .data import head

    return frozenset(w for r in head(500_000) for w in r["src"].split())


def _with_unknown(rng, src, held):
    """(order with something unknown put in, what was added). Three ways: a
    modifier before the pizza word ("a large hehe pizza"), the pizza word itself
    swapped for an off-menu thing ("a large horse with extra cheese"), or an
    extra clause ("... with zorblo")."""
    vocab = pizza_vocab()
    for _ in range(50):
        word = rng.choice([
            _nonsense_word(rng, held), _pool(rng, "adj", held), _pool(rng, "noun", held),
            _junk(rng, held).split()[0],
        ])
        if word not in vocab:
            break
    m = _HEAD.search(src)
    r = rng.random()
    if m and r < 0.4:
        return src[: m.start()] + word + " " + src[m.start():], word
    if m and r < 0.7:
        noun = _pool(rng, "noun", held)
        if noun not in vocab:
            return src[: m.start()] + noun + src[m.end():], noun
    return src.rstrip() + rng.choice([" with ", " and ", " plus "]) + word, word


def _row(kind, rng, held, base=None):
    """One negative. Whatever it adds to a real order must not be orderable."""
    for _ in range(50):
        if kind in ("mixed", "unknown"):
            if kind == "mixed":
                added = _off_phrase(rng, held)
                text = base["src"].rstrip() + rng.choice(JOIN) + added
            else:
                text, added = _with_unknown(rng, base["src"], held)
        else:
            text = added = _whole(kind, rng, held)
        if text and not on_menu(added):
            return {"src": text, "neg": kind}
    raise RuntimeError(f"could not generate an off-menu {kind} phrase")


def _base(kind, rng, rows):
    return rng.choice(rows) if kind in ("mixed", "unknown") else None


def _kinds(n, rng):
    names, weights = zip(*KINDS.items())
    return rng.choices(names, weights, k=n)


def target(row, arm):
    """The completion for a negative row. Ordinary rows never reach here."""
    if arm not in ARMS:
        raise ValueError(f"negatives need a ';'-separated arm {ARMS}, not {arm!r}")
    return REJECT


def mix(rows, frac, seed, arm):
    """`rows` plus enough negatives that they make up `frac` of the result.
    Built from training vocabulary and training orders only. Deterministic."""
    if not 0 < frac < 1:
        raise ValueError(f"negatives fraction must be in (0, 1), got {frac}")
    if arm not in ARMS:
        raise ValueError(f"negatives need a ';'-separated arm {ARMS}, not {arm!r}")
    rng = random.Random(20_000 + seed)
    n = round(len(rows) * frac / (1 - frac))
    out = list(rows)
    for kind in _kinds(n, rng):
        out.append(_row(kind, rng, held=False, base=_base(kind, rng, rows)))
    rng.shuffle(out)
    return out


def dev_rows(dev, n=120, seed=0):
    """For model selection during training: training vocabulary, dev orders."""
    rng = random.Random(25_000 + seed)
    return [_row(k, rng, held=False, base=_base(k, rng, dev)) for k in _kinds(n, rng)]


def eval_rows(test_rows, n=600, seed=0):
    """Held-out negatives: vocabulary never trained on, mixed with test orders."""
    rng = random.Random(30_000 + seed)
    return [_row(k, rng, held=True, base=_base(k, rng, test_rows)) for k in _kinds(n, rng)]


# A small challenge set of the kind of thing people type into a public demo,
# reported on every run. It is a smoke test, not the target: the generator, not
# this list, is what the model learns from.
PROBES = [
    "two cats and dog",
    "1 hello world cola",
    "lorem ipsum dolor sit amet consectetur adipiscing elit. quisque faucibus ex sapien "
    "vitae pellentesque sem placerat. in id cursus mi pretium tellus duis convallis. "
    "tempus leo eu aenean sed diam urna tempor. pulvinar vivamus fringilla lacus nec "
    "metus bibendum egestas. iaculis massa nisl malesuada lacinia integer nunc posuere. "
    "ut hendrerit semper vel class aptent taciti sociosqu. ad litora torquent per "
    "conubia nostra inceptos himenaeos.",
    "hehe hehe", "asdf", "lol", "🍕🍕🍕", "привет", "hello", "what is the meaning of life",
    "a dog with extra cheese", "two large horses", "i want a car", "blah blah blah",
]


def probe_rows():
    return [{"src": p, "neg": "probe"} for p in PROBES]


# ---------- scoring ----------


def is_reject(text):
    return text.strip() == REJECT


def correct(output, row, arm):
    """A negative is handled only by "?" alone: no ticket, nothing printed."""
    return is_reject(output)


def score(outputs, rows, arm, test_outputs=()):
    """Handled rate overall and per kind, plus how often a real test order is
    wrongly flagged (the price of strictness)."""
    by = {}
    for o, r in zip(outputs, rows):
        by.setdefault(r["neg"], []).append(correct(o, r, arm))
    res = {
        "n": len(rows),
        "handled": sum(sum(v) for v in by.values()) / max(1, len(rows)),
        **{f"handled_{k}": sum(v) / len(v) for k, v in sorted(by.items())},
    }
    if test_outputs:
        # Anything other than a ticket on a real order: "?" or "?" mixed in.
        res["false_reject_on_test"] = sum(REJECT in o.split() for o in test_outputs) / len(test_outputs)
    return res
