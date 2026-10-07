"""The menu vocabulary. Must stay in sync with the training notebook's generator."""

# Short code -> canonical drink name. The model emits the short codes.
DRINK_CODES = {
    "cap": "cappuccino",
    "lat": "latte",
    "ame": "americano",
    "esp": "espresso",
    "fw": "flat_white",
    "raf": "raf",
    "fil": "filter_coffee",
    "coc": "cocoa",
    "mat": "matcha_latte",
    "ice": "iced_latte",
}

SIZES = {"S", "M", "L"}
MILKS = {"oat", "almond", "coconut", "lactosefree", "nomilk"}
SYRUPS = {"vanilla", "caramel", "hazelnut"}

UNCLEAR = "?"
