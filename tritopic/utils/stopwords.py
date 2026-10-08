"""
Stopword Management for TriTopic
==================================

Provides language-aware stopword lists for TF-IDF vectorizers and keyword extraction.
Supports English, German, French, Spanish, and multilingual mode.
"""

from __future__ import annotations


# Minimal built-in stopword lists (no external dependency required).
# These cover the most frequent function words for each language.

_GERMAN_STOPWORDS = [
    "aber", "alle", "allem", "allen", "aller", "allerdings", "alles", "also",
    "am", "an", "ander", "andere", "anderem", "anderen", "anderer", "anderes",
    "anderm", "andern", "anders", "auch", "auf", "aus", "bei", "beim", "bereits",
    "bin", "bis", "bist", "bitte", "da", "dabei", "dadurch", "dafür", "dagegen",
    "daher", "dahin", "damals", "damit", "danach", "daneben", "dann", "daran",
    "darauf", "daraus", "darf", "darfst", "darin", "darum", "darunter", "das",
    "dass", "davon", "davor", "dazu", "dein", "deine", "deinem", "deinen",
    "deiner", "dem", "den", "denn", "dennoch", "der", "deren", "des", "deshalb",
    "dessen", "die", "dies", "diese", "dieselbe", "dieselben", "diesem", "diesen",
    "dieser", "dieses", "doch", "dort", "du", "durch", "dürfen", "ein", "einander",
    "eine", "einem", "einen", "einer", "einige", "einigem", "einigen", "einiger",
    "einiges", "einmal", "er", "erst", "es", "etwa", "etwas", "euch", "euer",
    "eure", "eurem", "euren", "eurer", "für", "ganz", "gar", "gegen", "gehen",
    "geht", "ging", "hab", "habe", "haben", "hat", "hatte", "hätte", "her",
    "herr", "hin", "hinter", "ich", "ihm", "ihn", "ihnen", "ihr", "ihre",
    "ihrem", "ihren", "ihrer", "im", "immer", "in", "indem", "infolge",
    "innen", "ins", "irgend", "ist", "ja", "jede", "jedem", "jeden", "jeder",
    "jedes", "jedoch", "jemals", "jene", "jenem", "jenen", "jener", "jenes",
    "jetzt", "kann", "kannst", "kein", "keine", "keinem", "keinen", "keiner",
    "kommen", "konnte", "können", "künftig", "lang", "lange", "längst", "längstens",
    "lässt", "laut", "lediglich", "machen", "macht", "man", "manch", "manche",
    "manchem", "manchen", "mancher", "manchmal", "mehr", "mein", "meine",
    "meinem", "meinen", "meiner", "mit", "muss", "müssen", "nach", "nachdem",
    "nachher", "nächst", "neben", "nein", "nicht", "nichts", "nie", "niemand",
    "noch", "nun", "nur", "ob", "oben", "oder", "ohne", "sehr", "seid", "sein",
    "seine", "seinem", "seinen", "seiner", "seit", "seitdem", "sich", "sie",
    "sind", "so", "sogar", "solch", "solche", "solchem", "solchen", "solcher",
    "soll", "sollen", "sollte", "sollten", "solltest", "sondern", "sonst",
    "soweit", "sowie", "über", "überhaupt", "übrigens", "um", "ums", "und",
    "uns", "unser", "unsere", "unserem", "unseren", "unserer", "unten", "unter",
    "viel", "viele", "vielem", "vielen", "vielleicht", "vom", "von", "vor",
    "vorbei", "vorher", "vorüber", "während", "wann", "war", "wäre", "warum",
    "was", "weder", "weil", "weit", "weiter", "weitere", "weiterem", "weiteren",
    "weiterer", "weiteres", "welch", "welche", "welchem", "welchen", "welcher",
    "welches", "wem", "wen", "wenig", "wenige", "wenigen", "weniger", "wenigstens",
    "wenn", "wer", "werde", "werden", "wessen", "wie", "wieder", "will", "wir",
    "wird", "wo", "wohl", "wollen", "worden", "wurde", "würde", "würden",
    "zu", "zum", "zur", "zwar", "zwischen",
]

_FRENCH_STOPWORDS = [
    "a", "ai", "aie", "aient", "aies", "ait", "alors", "après", "au", "aucun",
    "aura", "aurait", "aussi", "autre", "aux", "avaient", "avais", "avait",
    "avant", "avec", "avez", "aviez", "avions", "avoir", "avons", "ayant",
    "ayez", "ayons", "bien", "bon", "car", "ce", "ceci", "cela", "celle",
    "celles", "celui", "ces", "cet", "cette", "ceux", "chaque", "chez",
    "ci", "comme", "comment", "dans", "de", "dehors", "des", "deux",
    "devrait", "dire", "dit", "doit", "donc", "dont", "du", "elle",
    "elles", "en", "encore", "entre", "est", "et", "étaient", "était",
    "étant", "été", "être", "eu", "eue", "eues", "eurent", "eus",
    "eusse", "eussent", "eusses", "eussiez", "eussions", "eut", "eux",
    "faire", "fait", "faut", "fois", "font", "furent", "fus", "fusse",
    "fussent", "fusses", "fussiez", "fussions", "fut", "ici", "il", "ils",
    "je", "juste", "la", "le", "les", "leur", "leurs", "lui", "ma",
    "maintenant", "mais", "me", "même", "mes", "moi", "mon", "ne",
    "ni", "non", "nos", "notre", "nous", "on", "ont", "ou", "où",
    "par", "parce", "pas", "peut", "peu", "plus", "pour", "pourquoi",
    "quand", "que", "quel", "quelle", "quelles", "quels", "qui", "sa",
    "sans", "se", "sera", "serai", "seraient", "serais", "serait", "seras",
    "serez", "seriez", "serions", "serons", "seront", "ses", "si", "sien",
    "son", "sont", "soyez", "soyons", "suis", "sur", "ta", "te", "tes",
    "toi", "ton", "tous", "tout", "toute", "toutes", "très", "tu", "un",
    "une", "vos", "votre", "vous", "y",
]

_SPANISH_STOPWORDS = [
    "a", "al", "algo", "algunas", "algunos", "ambos", "ante", "antes",
    "aquí", "así", "aunque", "bien", "bueno", "cada", "como", "con",
    "contra", "cual", "cuando", "de", "del", "dentro", "desde", "donde",
    "dos", "el", "él", "ella", "ellas", "ellos", "en", "entre", "era",
    "esa", "esas", "ese", "eso", "esos", "esta", "está", "estaba",
    "estado", "están", "estar", "estas", "este", "esto", "estos",
    "fue", "fuera", "fueron", "ha", "había", "han", "hasta", "hay",
    "hoy", "ir", "la", "las", "le", "les", "lo", "los", "más", "me",
    "mejor", "menos", "mi", "mí", "mientras", "mis", "mismo", "mucho",
    "muy", "nada", "ni", "ningún", "ninguna", "ninguno", "no", "nos",
    "nosotros", "nuestro", "nueva", "nuevo", "o", "otra", "otro", "otros",
    "para", "pero", "poco", "por", "porque", "primero", "puede", "pues",
    "que", "qué", "quien", "quién", "se", "sea", "según", "ser", "si",
    "sí", "sido", "siempre", "sin", "sino", "sobre", "somos", "son",
    "soy", "su", "sus", "también", "tan", "tanto", "te", "tengo",
    "ti", "tiempo", "tiene", "toda", "todavía", "todo", "todos", "tres",
    "tú", "tu", "tus", "un", "una", "uno", "unos", "usted", "ustedes",
    "va", "vamos", "ver", "vez", "voy", "y", "ya", "yo",
]


_STOPWORD_LISTS: dict[str, list[str]] = {
    "german": _GERMAN_STOPWORDS,
    "french": _FRENCH_STOPWORDS,
    "spanish": _SPANISH_STOPWORDS,
}


# Token pattern for all vectorizers: a token starts with two letters and may
# continue with letters/digits (keeps "mr2", "covid19"; drops "000", "__",
# "1993", which otherwise pollute keywords of numeric-heavy corpora).
TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}[^\W_]*\b"


def get_stopwords(language: str) -> str | list[str] | None:
    """Return stopwords suitable for sklearn vectorizers.

    Parameters
    ----------
    language : str
        Language code: ``"english"``, ``"german"``, ``"french"``,
        ``"spanish"``, or ``"multilingual"``.

    Returns
    -------
    str | list[str] | None
        - ``"english"`` (sklearn built-in) for English
        - ``list[str]`` for German/French/Spanish
        - ``None`` for multilingual (no stopword filtering)
    """
    lang = language.lower().strip()

    if lang == "english":
        return "english"

    if lang == "multilingual":
        return None

    if lang in _STOPWORD_LISTS:
        return _STOPWORD_LISTS[lang]

    # Try nltk as fallback for other languages
    try:
        from nltk.corpus import stopwords as nltk_sw
        return nltk_sw.words(lang)
    except Exception:
        pass

    return None


def get_stopwords_set(language: str) -> set[str]:
    """Return stopwords as a ``set`` for fast membership checks.

    Always returns a set (possibly empty for multilingual / unknown).
    For English, expands sklearn's built-in list.
    """
    lang = language.lower().strip()

    if lang == "english":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        return set(ENGLISH_STOP_WORDS)

    if lang == "multilingual":
        return set()

    if lang in _STOPWORD_LISTS:
        return set(_STOPWORD_LISTS[lang])

    # Try nltk as fallback
    try:
        from nltk.corpus import stopwords as nltk_sw
        return set(nltk_sw.words(lang))
    except Exception:
        pass

    return set()
