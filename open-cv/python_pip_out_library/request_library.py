"""requests"""

# pip install requests

# import requests
# from pprint import pprint
#
# manzil= "https://kun.uz/news/main"
# r = requests.get(manzil)
# pprint(r.text)
# country = "Uzbekistan"
# url = f"https://restcountries.eu/rest/v2/name/{country}"
# r = requests.get(url)
# r_json = r.json()[0]
# print(r_json['capital'])


"""BeautifulSoup4"""
# pip install beautifulsoup4

"""BeautifulSoup juda kuchli modullardan biri bo'lib, bu modul yordamida turli veb sahifalardan istalgan ma'lumotlarni yig'ishtirib (scarpping) olish mumkin. Biror kishining instagram sahifasidagi barcha rasmlar deysizmi, Facebook guruhidagi barcha postlar va izohlar deysizmi, oldi-sotdi bozoridagi e'lonlar deysizmi, marhamat, bs4 moduli yordamida buni bemalol avtomatlashtirish mumkin.

Odatda bs4 moduli requests moduli bilan hamkorlikda ishlaydi. Keling, sodda misol kor'amiz. Avvalgi bo'limda, requests yordamida kun.uz sahifasining html kodini olgan edik. Endi esa bs4 yordamida html sahifadan oxirgi yangiliklarning mavzusini ajratib olamiz."""
import requests
from bs4 import BeautifulSoup
from pprint import pprint

sahifa = "https://kun.uz/news"
r = requests.get(sahifa)
pprint(r.text)

soup = BeautifulSoup(r.text, 'html.parser')
news = soup.find_all(class_="news-title") # yangiliklarning mavzusini ajratib olamiz
print(news) # eng birinchi yangilikni konsolga chiqaramiz