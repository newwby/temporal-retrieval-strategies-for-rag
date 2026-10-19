from __future__ import annotations

"""
fixed english word forms for years and days

not using the query phrasing cache (data/phrasing/phrasing_cache__llama3.1-8b.json) as it only
has the 995 query tokens, so it would KeyError on dates from the 32,678 corpus paragraphs.
filling it in with the llm would add model noise, so the language arm would measure the llm
rather than words vs digits

the cache also isn't consistent with itself, e.g.

    1280 -> "Twelve hundred and eighty"      1623 -> "sixteen twenty-three"
    1556 -> "One thousand five hundred..."   1880 -> "eighteen eighty"
    1928 -> "nineteen hundred twenty eight"  1940 -> "nineteen forty"

using it on the corpus would put that mix inside one arm. these rules use one style throughout:

    1997 -> nineteen ninety-seven      1900 -> nineteen hundred
    1905 -> nineteen oh five           1066 -> ten sixty-six
    2000 -> two thousand               2005 -> two thousand five
    2010 -> twenty ten                 2029 -> twenty twenty-nine

so the (language, language) grid cell isn't like for like, query side is llm worded and
corpus side is rule worded
"""

__all__ = ["day_to_words", "number_to_words", "year_to_words"]

_UNITS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen",
    "nineteen",
]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]

_ORDINALS = {
    1: "first", 2: "second", 3: "third", 5: "fifth", 8: "eighth", 9: "ninth", 12: "twelfth",
}


def number_to_words(value: int) -> str:
    if not 0 <= value <= 99:
        raise ValueError(f"number_to_words expects 0-99, got {value}")
    if value < 20:
        return _UNITS[value]
    tens, unit = divmod(value, 10)
    return _TENS[tens] if unit == 0 else f"{_TENS[tens]}-{_UNITS[unit]}"


"""
year as it's spoken, not as a plain number
read in pairs ("nineteen ninety-seven") except 2000-2009 ("two thousand n"), then pairs again ("twenty ten")
zero tens digit uses "oh" ("nineteen oh five"), zero second half uses "hundred" ("nineteen hundred")
"""
def year_to_words(year: int) -> str:
    if not 1000 <= year <= 2099:
        raise ValueError(f"year_to_words expects 1000-2099, got {year}")

    century, remainder = divmod(year, 100)

    if 2000 <= year <= 2009:
        return "two thousand" if remainder == 0 else f"two thousand {_UNITS[remainder]}"

    century_words = number_to_words(century)
    if remainder == 0:
        return f"{century_words} hundred"
    if remainder < 10:
        return f"{century_words} oh {_UNITS[remainder]}"
    return f"{century_words} {number_to_words(remainder)}"


def day_to_words(day: int) -> str:
    if not 1 <= day <= 31:
        raise ValueError(f"day_to_words expects 1-31, got {day}")
    if day in _ORDINALS:
        return _ORDINALS[day]
    if day < 20:
        return f"{_UNITS[day]}th"
    tens, unit = divmod(day, 10)
    if unit == 0:
        return f"{_TENS[tens][:-1]}ieth"
    return f"{_TENS[tens]}-{_ORDINALS.get(unit, _UNITS[unit] + 'th')}"
