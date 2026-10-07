# Frozen copy of the inline generator from coffee_order_finetune.ipynb as it stood when
# the production v2 model was trained. Do not edit: tests assert that
# coffee_parser.generator.production_dataset() still reproduces this exactly.
import random
random.seed(42)

# code -> spoken variants. Codes must match coffee_eval.py
# (флэтик and капучиночку are deliberately left out: the test set checks generalization)
DRINKS = {
  "cap": {"ru":["капучино","капуччино","капучина","капчино","капучинку","капучинка"], "en":["cappuccino","capuccino","cappuccinos"]},
  "lat": {"ru":["латте","лате","латэ","латтешку","латтешка"], "en":["latte","lattes","late"]},
  "ame": {"ru":["американо","американа","американчик"], "en":["americano","americanos"]},
  "esp": {"ru":["эспрессо","экспрессо","эспресо","эспрессик"], "en":["espresso","expresso","espressos"]},
  "fw":  {"ru":["флэт уайт","флет вайт","флэтуайт"], "en":["flat white","flat whites","flatwhite"]},
  "raf": {"ru":["раф","раф кофе","рафик"], "en":["raf","raf coffee"]},
  "fil": {"ru":["фильтр кофе","фильтр","черный кофе","кофе из фильтра"], "en":["filter coffee","drip coffee","black coffee"]},
  "coc": {"ru":["какао","горячий шоколад"], "en":["cocoa","hot chocolate"]},
  "mat": {"ru":["матча латте","матча","матчу"], "en":["matcha latte","matcha"]},
  "ice": {"ru":["айс латте","холодный латте","латте со льдом"], "en":["iced latte","ice latte","cold latte"]},
}
SIZES = {"S":{"ru":["маленький","маленькую","маленькое","поменьше"],"en":["small"]},
         "M":{"ru":["средний","среднюю","стандартный","обычный"],"en":["medium","regular"]},
         "L":{"ru":["большой","большую","большое","побольше"],"en":["large","big"]}}
MILKS = {"oat":{"ru":["на овсяном","на овсяном молоке","с овсяным молоком"],"en":["with oat milk","oat"]},
         "almond":{"ru":["на миндальном","с миндальным молоком"],"en":["with almond milk","almond"]},
         "coconut":{"ru":["на кокосовом","на кокосовом молоке"],"en":["with coconut milk","coconut"]},
         "lactosefree":{"ru":["на безлактозном","на безлактозном молоке"],"en":["lactose free","with lactose free milk"]},
         "nomilk":{"ru":["без молока"],"en":["no milk","without milk"]}}
SYRUPS = {"vanilla":{"ru":["с ванильным сиропом","с ванилью","ванильный"],"en":["with vanilla syrup","vanilla"]},
          "caramel":{"ru":["с карамелью","с карамельным сиропом","карамельный"],"en":["with caramel","caramel"]},
          "hazelnut":{"ru":["с ореховым сиропом","с лесным орехом","ореховый"],"en":["with hazelnut","hazelnut"]}}
QTY = {"ru":{1:["","один","одну","одно"],2:["два","две","2"],3:["три","3"]},
       "en":{1:["","one","a","an"],2:["two","2"],3:["three","3"]}}
NOISE = {"ru":["с собой","на вынос","здесь","без сахара","погорячее","пожалуйста"],
         "en":["to go","for here","no sugar","please"]}
FILLERS = {"ru":["здравствуйте","мне пожалуйста","можно","дайте","я бы хотел","будьте добры","эээ","ну","добрый день",""],
           "en":["hi","can i get","i'd like","let me get","um","could i have",""]}
AND = {"ru":[" и ", " еще ", " и еще ", " плюс "], "en":[" and ", " also ", " plus "]}
FIX = {"ru":["нет лучше","ой нет давайте","точнее","нет подождите"], "en":["actually scratch that","no wait make it","sorry i mean"]}
OFF_MENU = {"ru":["черный чай","зеленый чай","чай","круассан","чизкейк","сэндвич","пиво","лимонад","смузи","бутылку воды","мятный раф","кофе с корицей"],
            "en":["tea","croissant","muffin","sandwich","beer","a bottle of water","smoothie"]}
NON_ORDERS = {"ru":["где туалет","а туалет где","какой пароль от вайфая","можно картой","вы до скольки работаете","что посоветуете",
                    "как всегда","мне как в прошлый раз","что-нибудь горячее","просто кофе","какой-нибудь кофе","у вас есть розетка",
                    "сколько стоит {d}","а {d} вкусный","спасибо большое","хорошего дня","подождите секунду","я еще думаю"],
              "en":["where is the restroom","what's the wifi password","can i pay by card","what do you recommend","just coffee",
                    "the usual","how much is the {d}","thank you","have a nice day","give me a second"]}
NO_MILK_DRINKS = {"esp","ame","fil"}

def typo(w, p=0.05):
    if len(w) > 4 and random.random() < p:
        i = random.randrange(1, len(w)-1)
        return w[:i] + w[i+1:]
    return w

def code_of(d, size=None, milk=None, syrup=None, qty=1):
    parts = [d] + [x for x in (size, milk, syrup) if x]
    if qty > 1: parts.append(f"x{qty}")
    return " ".join(parts)

def make_item(lang):
    d = random.choice(list(DRINKS))
    qty = random.choices([1,2,3],[0.8,0.15,0.05])[0]
    size = random.choice(list(SIZES)) if random.random() < 0.6 else None
    milk = random.choice(list(MILKS)) if d not in NO_MILK_DRINKS and random.random() < 0.35 else None
    syrup = random.choice(list(SYRUPS)) if random.random() < 0.2 else None
    mods = []
    if size: mods.append(random.choice(SIZES[size][lang]))
    if milk: mods.append(random.choice(MILKS[milk][lang]))
    if syrup: mods.append(random.choice(SYRUPS[syrup][lang]))
    if random.random() < 0.15: mods.append(random.choice(NOISE[lang]))
    random.shuffle(mods)
    split = random.randint(0, len(mods))  # some modifiers before the drink, some after
    q = random.choice(QTY[lang][qty])
    words = ([q] if q else []) + mods[:split] + [random.choice(DRINKS[d][lang])] + mods[split:]
    text = " ".join(words)
    if size and random.random() < 0.06:  # size correction: "большой ой нет средний"
        new = random.choice([s for s in SIZES if s != size])
        text += " " + random.choice(FIX[lang]) + " " + random.choice(SIZES[new][lang])
        size = new
    return code_of(d, size, milk, syrup, qty), text

def split_pair(lang):
    # "два капучино один на миндальном" -> cap almond;cap
    d = random.choice([k for k in DRINKS if k not in NO_MILK_DRINKS])
    milk = random.choice(list(MILKS))
    if lang == "ru":
        text = f"{random.choice(['два','две','2'])} {random.choice(DRINKS[d]['ru'])} {random.choice(['один','одну','одно'])} {random.choice(MILKS[milk]['ru'])}"
    else:
        text = f"two {random.choice(DRINKS[d]['en'])} one {random.choice(MILKS[milk]['en'])}"
    return f"{d} {milk};{d}", text

def finish(text, lang):
    text = (random.choice(FILLERS[lang]) + " " + text).strip()
    return " ".join(typo(w) for w in text.lower().split())

def merge(outs):
    # "fil;fil x2" -> "fil x3" so the model learns one consistent answer
    order, qty = [], {}
    for o in outs:
        toks = o.split()
        q = int(toks[-1][1:]) if toks[-1].startswith("x") else 1
        key = " ".join(t for t in toks if not (t.startswith("x") and t[1:].isdigit()))
        if key not in qty: order.append(key); qty[key] = 0
        qty[key] += q
    return [k if k == "?" or qty[k] == 1 else f"{k} x{qty[k]}" for k in order]

def make_example():
    lang = random.choices(["ru","en"],[0.75,0.25])[0]
    r = random.random()
    if r < 0.12:   # not an order
        t = random.choice(NON_ORDERS[lang]).replace("{d}", random.choice(DRINKS[random.choice(list(DRINKS))][lang]))
        return {"input": finish(t, lang) if random.random() < 0.3 else t, "output": "?"}
    if r < 0.18:   # only off-menu items
        return {"input": finish(random.choice(OFF_MENU[lang]), lang), "output": "?"}
    if r < 0.23:   # same drink, different modifiers
        out, text = split_pair(lang)
        return {"input": finish(text, lang), "output": out}
    n = random.choices([1,2,3],[0.7,0.22,0.08])[0]
    outs, texts = zip(*[make_item(lang) for _ in range(n)])
    outs, text = list(outs), texts[0]
    for t in texts[1:]:
        text += random.choice(AND[lang]) + t
    if r < 0.29:   # order + an off-menu item -> known items + "?"
        text += random.choice(AND[lang]) + random.choice(OFF_MENU[lang])
        outs.append("?")
    if random.random() < 0.08:  # customer changes their mind
        _, wrong = make_item(lang)
        text = wrong + " " + random.choice(FIX[lang]) + " " + text
    return {"input": finish(text, lang), "output": ";".join(merge(outs))}

data = [make_example() for _ in range(4000)]
random.shuffle(data)
train_data, val_data = data[:3800], data[3800:]
