# """wordcloud va matplotlib"""
#
# # pip install wordcloud
# """Wordcloud moduli yordamida katta matnlarda eng ko'p uchraydigan so'zlarni chiroyli qilib, so'zlar buluti chiqarish mumkin. 2020-yil yakunida, sariqdev sahifasida chop etilgan mashxur blogerlarning siluetlari ham aynan shu modul yordamida qilingan."""
# # pip install matplotlib
#
# import requests
#
# from bs4 import BeautifulSoup
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt
#
# sahifa = "https://kun.uz/news"
# r = requests.get(sahifa)
#
# soup = BeautifulSoup(r.text, 'html.parser')
# news = soup.find_all(class_="news-title")
# matn = ""
# for n in news:
#     matn += n.text
#
# # kerakmas so'zlar
# stopwords = ["учун", "бўйича", "лекин", "билан", "ва", "бор", "ҳам", "хил", "йил", "ва", "ёки", "билан", "ҳам", "деб", "учун", "бу", "шу", "у", "эса"]# bulutni yaratamiz
# wordcloud = WordCloud(width=1000, height=1000,
#                       background_color='white',
#                       stopwords=stopwords,
#                       min_font_size=20).generate(matn)
#
# # plot the WordCloud image
# plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()



import requests
from bs4 import BeautifulSoup
from wordcloud import WordCloud
import matplotlib.pyplot as plt

sahifa = "https://kun.uz/news/main"
r = requests.get(sahifa)

soup = BeautifulSoup(r.text, 'html.parser')

try:
    news = soup.find_all(class_="news-title")
except:
    print("Veb-sayt tuzilishi o'zgargan")
    news = []

matn = ' '.join([n.text for n in news])

stopwords = ["учун", "бўйича", "лекин", "билан", "ва", "бор", "ҳам", "хил", "йил", "ва", "ёки", "билан", "ҳам", "деб", "учун", "бу", "шу", "у", "эса"]

wordcloud = WordCloud(width=1000, height=1000,
                      background_color='white',
                      stopwords=stopwords,
                      min_font_size=20).generate(matn)

plt.figure(figsize=(12, 12), facecolor='grey')
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()