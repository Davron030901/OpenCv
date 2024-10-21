"""fuzzywuzzy"""

# pip install fuzzywuzzy
"""
Bu modul yordamida so'zlarni bir-biriga solishtirish yoki matnlar tarkibida kerakli so'zni topishda foydalanamiz.

Quyidagi misolda "salom" so'zini "assalom" va "salim" so'zlari bilan naqadar o'xshashligini tekshrib ko'ramiz:
# """
# from fuzzywuzzy import fuzz
# print(fuzz.ratio("salom",'assalom'))
#
# print(fuzz.ratio("salom","salim"))

from fuzzywuzzy import process
from uzwords import words

# Matnlar orasidan 3 ta eng o'xshashlarini ajratib olish
text = "салом"
matches = process.extract(text, words, limit=3)
print(matches)
[('салом', 100), ('салдом', 91), ('слалом', 91)]
# Matnlar orasidan eng o'xshashini topish
text = "талба"
match = process.extractOne(text,words)
print(match)