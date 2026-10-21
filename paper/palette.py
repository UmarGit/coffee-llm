"""One palette, used by the matplotlib figures and the TikZ illustrations."""
ARMS = ["exr", "json", "exr_short", "tagged", "dsl", "words"]
LABEL = {"exr": "EXR", "json": "JSON", "exr_short": "EXR-short",
         "tagged": "TAGGED", "dsl": "DSL", "words": "WORDS"}
MACRO = {"exr": "exr", "json": "json", "exr_short": "exrShort",
         "tagged": "tagged", "dsl": "dsl", "words": "words"}
COLOR = {"exr": "#2D4A6B", "json": "#5B8FB0", "exr_short": "#7A9A5A",
         "tagged": "#D99A2B", "dsl": "#C4513B", "words": "#8E5A9E"}
SEMANTIC = {"correct": "#CFD8CF", "silent": "#C4513B", "detectable": "#E8C48A",
            "ink": "#22303F", "muted": "#8A94A0", "rule": "#D7DCE2"}


def latex_colors():
    out = []
    for name, hexv in list(COLOR.items()) + list(SEMANTIC.items()):
        key = MACRO.get(name, name)
        out.append(f"\\definecolor{{c{key[0].upper()}{key[1:]}}}{{HTML}}{{{hexv.lstrip('#')}}}")
    return "\n".join(out)
