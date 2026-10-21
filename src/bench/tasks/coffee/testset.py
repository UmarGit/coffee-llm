"""Hand-written evaluation set: (spoken text, expected compact answer).

Written separately from the training generator. Some slang (флэтик, капучиночку)
is deliberately NOT in the generator, to test generalization.
"""

from .menu import UNCLEAR

TEST_SET = [
    # plain orders
    ("эспрессо", "esp"),
    ("ну давайте капучино", "cap"),
    ("можно пожалуйста айс латте", "ice"),
    ("фильтр кофе средний", "fil M"),
    ("какао для ребенка маленькое", "coc S"),
    ("мне американчик маленький", "ame S"),
    ("черный кофе без сахара", "fil"),
    # diminutives / slang
    ("добрый день можно капучинку побольше", "cap L"),
    ("латтешку на овсяном пожалуйста", "lat oat"),
    ("капучиночку", "cap"),
    ("флэтик на кокосовом", "fw coconut"),
    # modifiers
    ("раф ванильный большой", "raf L vanilla"),
    ("а можно флэт уайт на кокосовом молоке", "fw coconut"),
    ("матчу на овсяном большую", "mat L oat"),
    ("холодный латте с карамелью", "ice caramel"),
    ("капучино без сахара на безлактозном", "cap lactosefree"),
    ("раф на кокосовом с карамелью маленький", "raf S coconut caramel"),
    ("капучино с собой средний", "cap M"),
    # typos (speech-to-text errors)
    ("амерникано", "ame"),
    ("латэ большой", "lat L"),
    ("капучина на миндальном", "cap almond"),
    # quantities
    ("три латте с собой", "lat x3"),
    ("2 американо и 1 эспрессо", "ame x2;esp"),
    ("мне эспрессо и еще один эспрессо", "esp x2"),
    # multiple items
    (
        "здравствуйте мне пожалуйста латте с ореховым сиропом и маленький американо",
        "lat hazelnut;ame S",
    ),
    ("капучино средний на миндальном и какао большое", "cap M almond;coc L"),
    ("два капучино один на миндальном", "cap almond;cap"),
    ("два раф один с ванилью другой с карамелью", "raf vanilla;raf caramel"),
    # corrections
    ("капучино нет подождите латте", "lat"),
    ("большой латте ой нет средний", "lat M"),
    # partly off-menu -> known items + "?"
    ("капучино и круассан", "cap;?"),
    ("лавандовый раф", "raf;?"),
    # not an order / off-menu / too vague -> "?"
    ("где у вас туалет", "?"),
    ("сколько стоит латте", "?"),
    ("есть вайфай", "?"),
    ("спасибо до свидания", "?"),
    ("чай с лимоном", "?"),
    ("круассан", "?"),
    ("то же самое что вчера", "?"),
    ("мне как обычно", "?"),
    ("кофе", "?"),
    ("оплата картой", "?"),
    # English
    ("can i get a large oat latte", "lat L oat"),
    ("two flat whites please", "fw x2"),
    ("small cappuccino no wait make it medium", "cap M"),
    ("hot chocolate for my kid", "coc"),
    ("an iced latte with vanilla and an americano", "ice vanilla;ame"),
    ("where's the bathroom", "?"),
    ("green tea please", "?"),
]


def category(expected):
    """Bucket a case for the per-category breakdown."""
    if expected == UNCLEAR:
        return "not-order"
    if UNCLEAR in expected:
        return "partial"
    return "order"
