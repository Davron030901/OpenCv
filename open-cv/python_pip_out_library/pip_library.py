"""pip"""

"""googletrans"""

# pip install googletrans==3.1.0a0

from googletrans import Translator
tarjimon = Translator() # Translator bu maxsus klass (tarjimon esa obyekt)
matn_uz = "Python - dunyodagi eng mashxur dasturlash tili"
tarjima = tarjimon.translate(matn_uz)
print(tarjima.text)

# pip uninstall googletrans # modulni o'chirish
#
# pip install googletrans==3.1.0a0 # yangi verisyani o'rnatish

tarjima_ru = tarjimon.translate(matn_uz, dest='ru')
print(tarjima_ru.text)

matn_en = "Tashkent is the capital of Uzbekistan"
tarjima_uz = tarjimon.translate(matn_en, dest='uz')
print(tarjima_uz.text)

tarjima_uz = tarjimon.translate(matn_en, src='uz', dest='uz')
print(tarjima_uz.text)

print(tarjima.origin)

print(tarjima.text)

print(tarjima.src)