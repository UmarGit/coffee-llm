"""Synthetic order generator.

Deterministic given a seed. `Difficulty.DEFAULT` reproduces the distribution used to
train the production model; `Difficulty.HARD` raises item counts, typos and corrections
for an eval split that is not at the accuracy ceiling.

Spoken variants live here; the output vocabulary lives in menu.py. The two are kept
consistent by `assert_menu_consistent()`.
"""

import random
from dataclasses import dataclass

from .menu import (
    DRINK_CODES,
)
from .menu import (
    MILKS as MENU_MILKS,
)
from .menu import (
    SIZES as MENU_SIZES,
)
from .menu import (
    SYRUPS as MENU_SYRUPS,
)

DRINKS = {
    "cap": {
        "ru": [
            "капучино",
            "капуччино",
            "капучина",
            "капчино",
            "капучинку",
            "капучинка",
        ],
        "en": ["cappuccino", "capuccino", "cappuccinos"],
    },
    "lat": {
        "ru": ["латте", "лате", "латэ", "латтешку", "латтешка"],
        "en": ["latte", "lattes", "late"],
    },
    "ame": {
        "ru": ["американо", "американа", "американчик"],
        "en": ["americano", "americanos"],
    },
    "esp": {
        "ru": ["эспрессо", "экспрессо", "эспресо", "эспрессик"],
        "en": ["espresso", "expresso", "espressos"],
    },
    "fw": {
        "ru": ["флэт уайт", "флет вайт", "флэтуайт"],
        "en": ["flat white", "flat whites", "flatwhite"],
    },
    "raf": {"ru": ["раф", "раф кофе", "рафик"], "en": ["raf", "raf coffee"]},
    "fil": {
        "ru": ["фильтр кофе", "фильтр", "черный кофе", "кофе из фильтра"],
        "en": ["filter coffee", "drip coffee", "black coffee"],
    },
    "coc": {"ru": ["какао", "горячий шоколад"], "en": ["cocoa", "hot chocolate"]},
    "mat": {"ru": ["матча латте", "матча", "матчу"], "en": ["matcha latte", "matcha"]},
    "ice": {
        "ru": ["айс латте", "холодный латте", "латте со льдом"],
        "en": ["iced latte", "ice latte", "cold latte"],
    },
}
SIZES = {
    "S": {"ru": ["маленький", "маленькую", "маленькое", "поменьше"], "en": ["small"]},
    "M": {
        "ru": ["средний", "среднюю", "стандартный", "обычный"],
        "en": ["medium", "regular"],
    },
    "L": {"ru": ["большой", "большую", "большое", "побольше"], "en": ["large", "big"]},
}
MILKS = {
    "oat": {
        "ru": ["на овсяном", "на овсяном молоке", "с овсяным молоком"],
        "en": ["with oat milk", "oat"],
    },
    "almond": {
        "ru": ["на миндальном", "с миндальным молоком"],
        "en": ["with almond milk", "almond"],
    },
    "coconut": {
        "ru": ["на кокосовом", "на кокосовом молоке"],
        "en": ["with coconut milk", "coconut"],
    },
    "lactosefree": {
        "ru": ["на безлактозном", "на безлактозном молоке"],
        "en": ["lactose free", "with lactose free milk"],
    },
    "nomilk": {"ru": ["без молока"], "en": ["no milk", "without milk"]},
}
SYRUPS = {
    "vanilla": {
        "ru": ["с ванильным сиропом", "с ванилью", "ванильный"],
        "en": ["with vanilla syrup", "vanilla"],
    },
    "caramel": {
        "ru": ["с карамелью", "с карамельным сиропом", "карамельный"],
        "en": ["with caramel", "caramel"],
    },
    "hazelnut": {
        "ru": ["с ореховым сиропом", "с лесным орехом", "ореховый"],
        "en": ["with hazelnut", "hazelnut"],
    },
}
QTY = {
    "ru": {
        1: ["", "один", "одну", "одно"],
        2: ["два", "две", "2"],
        3: ["три", "3"],
        4: ["четыре", "4"],
    },
    "en": {
        1: ["", "one", "a", "an"],
        2: ["two", "2"],
        3: ["three", "3"],
        4: ["four", "4"],
    },
}
NOISE = {
    "ru": ["с собой", "на вынос", "здесь", "без сахара", "погорячее", "пожалуйста"],
    "en": ["to go", "for here", "no sugar", "please"],
}
FILLERS = {
    "ru": [
        "здравствуйте",
        "мне пожалуйста",
        "можно",
        "дайте",
        "я бы хотел",
        "будьте добры",
        "эээ",
        "ну",
        "добрый день",
        "",
    ],
    "en": ["hi", "can i get", "i'd like", "let me get", "um", "could i have", ""],
}
AND = {"ru": [" и ", " еще ", " и еще ", " плюс "], "en": [" and ", " also ", " plus "]}
FIX = {
    "ru": ["нет лучше", "ой нет давайте", "точнее", "нет подождите"],
    "en": ["actually scratch that", "no wait make it", "sorry i mean"],
}
OFF_MENU = {
    "ru": [
        "черный чай",
        "зеленый чай",
        "чай",
        "круассан",
        "чизкейк",
        "сэндвич",
        "пиво",
        "лимонад",
        "смузи",
        "бутылку воды",
        "мятный раф",
        "кофе с корицей",
    ],
    "en": [
        "tea",
        "croissant",
        "muffin",
        "sandwich",
        "beer",
        "a bottle of water",
        "smoothie",
    ],
}
NON_ORDERS = {
    "ru": [
        "где туалет",
        "а туалет где",
        "какой пароль от вайфая",
        "можно картой",
        "вы до скольки работаете",
        "что посоветуете",
        "как всегда",
        "мне как в прошлый раз",
        "что-нибудь горячее",
        "просто кофе",
        "какой-нибудь кофе",
        "у вас есть розетка",
        "сколько стоит {d}",
        "а {d} вкусный",
        "спасибо большое",
        "хорошего дня",
        "подождите секунду",
        "я еще думаю",
    ],
    "en": [
        "where is the restroom",
        "what's the wifi password",
        "can i pay by card",
        "what do you recommend",
        "just coffee",
        "the usual",
        "how much is the {d}",
        "thank you",
        "have a nice day",
        "give me a second",
    ],
}
NO_MILK_DRINKS = {"esp", "ame", "fil"}


@dataclass(frozen=True)
class Difficulty:
    item_counts: tuple = (1, 2, 3)
    item_weights: tuple = (0.7, 0.22, 0.08)
    typo_rate: float = 0.05
    correction_rate: float = 0.08
    size_correction_rate: float = 0.06
    size_rate: float = 0.6
    milk_rate: float = 0.35
    syrup_rate: float = 0.2
    noise_rate: float = 0.15
    en_rate: float = 0.25
    p_non_order: float = 0.12
    p_off_menu: float = 0.18
    p_split_pair: float = 0.23
    p_order_plus_off_menu: float = 0.29


DEFAULT = Difficulty()

# More items per order, far more typos, corrections and noise. Same label semantics.
HARD = Difficulty(
    item_counts=(1, 2, 3, 4),
    item_weights=(0.3, 0.3, 0.25, 0.15),
    typo_rate=0.18,
    correction_rate=0.3,
    size_correction_rate=0.2,
    size_rate=0.75,
    milk_rate=0.5,
    syrup_rate=0.35,
    noise_rate=0.4,
    p_order_plus_off_menu=0.34,
)


def _typo(rng, w, p):
    if len(w) > 4 and rng.random() < p:
        i = rng.randrange(1, len(w) - 1)
        return w[:i] + w[i + 1 :]
    return w


def code_of(d, size=None, milk=None, syrup=None, qty=1):
    parts = [d] + [x for x in (size, milk, syrup) if x]
    if qty > 1:
        parts.append(f"x{qty}")
    return " ".join(parts)


def _make_item(rng, lang, cfg):
    d = rng.choice(list(DRINKS))
    qty = rng.choices([1, 2, 3], [0.8, 0.15, 0.05])[0]
    size = rng.choice(list(SIZES)) if rng.random() < cfg.size_rate else None
    milk = (
        rng.choice(list(MILKS))
        if d not in NO_MILK_DRINKS and rng.random() < cfg.milk_rate
        else None
    )
    syrup = rng.choice(list(SYRUPS)) if rng.random() < cfg.syrup_rate else None
    mods = []
    if size:
        mods.append(rng.choice(SIZES[size][lang]))
    if milk:
        mods.append(rng.choice(MILKS[milk][lang]))
    if syrup:
        mods.append(rng.choice(SYRUPS[syrup][lang]))
    if rng.random() < cfg.noise_rate:
        mods.append(rng.choice(NOISE[lang]))
    rng.shuffle(mods)
    split = rng.randint(0, len(mods))
    q = rng.choice(QTY[lang][qty])
    words = (
        ([q] if q else []) + mods[:split] + [rng.choice(DRINKS[d][lang])] + mods[split:]
    )
    text = " ".join(words)
    if size and rng.random() < cfg.size_correction_rate:
        new = rng.choice([s for s in SIZES if s != size])
        text += " " + rng.choice(FIX[lang]) + " " + rng.choice(SIZES[new][lang])
        size = new
    return code_of(d, size, milk, syrup, qty), text


def _split_pair(rng, lang):
    d = rng.choice([k for k in DRINKS if k not in NO_MILK_DRINKS])
    milk = rng.choice(list(MILKS))
    if lang == "ru":
        text = f"{rng.choice(['два', 'две', '2'])} {rng.choice(DRINKS[d]['ru'])} {rng.choice(['один', 'одну', 'одно'])} {rng.choice(MILKS[milk]['ru'])}"
    else:
        text = f"two {rng.choice(DRINKS[d]['en'])} one {rng.choice(MILKS[milk]['en'])}"
    return f"{d} {milk};{d}", text


def _finish(rng, text, lang, cfg):
    text = (rng.choice(FILLERS[lang]) + " " + text).strip()
    return " ".join(_typo(rng, w, cfg.typo_rate) for w in text.lower().split())


def _merge(outs):
    order, qty = [], {}
    for o in outs:
        toks = o.split()
        q = int(toks[-1][1:]) if toks[-1].startswith("x") else 1
        key = " ".join(t for t in toks if not (t.startswith("x") and t[1:].isdigit()))
        if key not in qty:
            order.append(key)
            qty[key] = 0
        qty[key] += q
    return [k if k == "?" or qty[k] == 1 else f"{k} x{qty[k]}" for k in order]


def make_example(rng, cfg=DEFAULT):
    lang = rng.choices(["ru", "en"], [1 - cfg.en_rate, cfg.en_rate])[0]
    r = rng.random()
    if r < cfg.p_non_order:
        t = rng.choice(NON_ORDERS[lang]).replace(
            "{d}", rng.choice(DRINKS[rng.choice(list(DRINKS))][lang])
        )
        return {
            "input": _finish(rng, t, lang, cfg) if rng.random() < 0.3 else t,
            "output": "?",
        }
    if r < cfg.p_off_menu:
        return {
            "input": _finish(rng, rng.choice(OFF_MENU[lang]), lang, cfg),
            "output": "?",
        }
    if r < cfg.p_split_pair:
        out, text = _split_pair(rng, lang)
        return {"input": _finish(rng, text, lang, cfg), "output": out}
    n = rng.choices(list(cfg.item_counts), list(cfg.item_weights))[0]
    outs, texts = zip(*[_make_item(rng, lang, cfg) for _ in range(n)])
    outs, text = list(outs), texts[0]
    for t in texts[1:]:
        text += rng.choice(AND[lang]) + t
    if r < cfg.p_order_plus_off_menu:
        text += rng.choice(AND[lang]) + rng.choice(OFF_MENU[lang])
        outs.append("?")
    if rng.random() < cfg.correction_rate:
        _, wrong = _make_item(rng, lang, cfg)
        text = wrong + " " + rng.choice(FIX[lang]) + " " + text
    return {"input": _finish(rng, text, lang, cfg), "output": ";".join(_merge(outs))}


def generate(n, seed, cfg=DEFAULT, shuffle=True):
    """n examples, deterministic in (n, seed, cfg)."""
    rng = random.Random(seed)
    data = [make_example(rng, cfg) for _ in range(n)]
    if shuffle:
        rng.shuffle(data)
    return data


def generate_unique(n, seed, cfg=DEFAULT, exclude=(), max_tries=60):
    """n examples with distinct inputs, none of them in `exclude`."""
    seen, out, rng = set(exclude), [], random.Random(seed)
    for _ in range(max_tries * n):
        ex = make_example(rng, cfg)
        if ex["input"] in seen or not ex["input"].strip():
            continue
        seen.add(ex["input"])
        out.append(ex)
        if len(out) == n:
            return out
    raise RuntimeError(
        f"only produced {len(out)}/{n} unique examples; widen the generator or lower n"
    )


def assert_menu_consistent():
    assert set(DRINKS) == set(DRINK_CODES), set(DRINKS) ^ set(DRINK_CODES)
    assert set(SIZES) == MENU_SIZES, set(SIZES) ^ MENU_SIZES
    assert set(MILKS) == MENU_MILKS, set(MILKS) ^ MENU_MILKS
    assert set(SYRUPS) == MENU_SYRUPS, set(SYRUPS) ^ MENU_SYRUPS


def production_dataset():
    """Exactly the dataset the production notebook trained on."""
    data = generate(4000, seed=42)
    return data[:3800], data[3800:]
