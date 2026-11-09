"""One palette for the matplotlib figures and the LaTeX colours.

Okabe-Ito hues (colour-blind safe). Every format also has its own marker and line
style, so a figure still reads in greyscale.
"""
ARMS = ["exr", "json", "exr_short", "tagged", "dsl", "words"]
LABEL = {"exr": "EXR", "json": "JSON", "exr_short": "EXR-short",
         "tagged": "TAGGED", "dsl": "DSL", "words": "WORDS"}
MACRO = {"exr": "exr", "json": "json", "exr_short": "exrShort",
         "tagged": "tagged", "dsl": "dsl", "words": "words"}
COLOR = {"exr": "#0072B2", "json": "#56B4E9", "exr_short": "#009E73",
         "tagged": "#E69F00", "dsl": "#D55E00", "words": "#000000"}
MARKER = {"exr": "o", "json": "s", "exr_short": "^", "tagged": "D", "dsl": "v", "words": "*"}
LINE = {"exr": "-", "json": (0, (4, 1.5)), "exr_short": (0, (1, 1.2)),
        "tagged": (0, (4, 1.2, 1, 1.2)), "dsl": (0, (2, 1)), "words": "-"}
OUTCOME = {"correct": ("#E8E8E8", ""), "silent": ("#D55E00", ""),
           "d_parse": ("#F0E442", "////"), "d_refuse": ("#56B4E9", "....")}
SEMANTIC = {"ink": "#22303F", "muted": "#7A7A7A", "rule": "#D7DCE2",
            "correct": "#E8E8E8", "silent": "#D55E00"}


def latex_colors():
    out = []
    for name, hexv in list(COLOR.items()) + list(SEMANTIC.items()):
        key = MACRO.get(name, name)
        out.append(f"\\definecolor{{c{key[0].upper()}{key[1:]}}}{{HTML}}{{{hexv.lstrip('#')}}}")
    return "\n".join(out)
