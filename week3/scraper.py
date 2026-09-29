import requests
from bs4 import BeautifulSoup

headers = {"User-Agent": "art-python-course/1.0"}

url = "https://en.wikipedia.org/wiki/Sol_LeWitt"

response = requests.get(url, headers=headers)

print(response.status_code)

print(len(response.text), "символов")
print(response.text[:400])
soup = BeautifulSoup(response.text,"html.parser")

print(soup.title.text)

paragraphs = soup.find_all("p")
for p in paragraphs[:5]:
        print(p.get_text().strip()[:200])
        print('----'*8)


image_urls=[]
for img in soup.find_all('img'):
    src = img.get("src","")
    if ".jpg" in src.lower():
        if src.startswith ("//"):
            src = "https:"+src
        image_urls.append(src)

#links = []
#for a in soup.find_all("a"):
    #if a.get("href", "").startswith('/wiki/'):
        # href = "https://wikipedia.org" + a.get("href")
        #links.append(href)

links_alt = []
for a in soup.find_all("a"):
    href = a.get("href", "")
    if href.startswith("https://en.")\
         or href.startswith("https://ru."):
         links_alt.append(href)

print(f"ссылок на статьи: {len(links_alt)}")


links_alt_txt = '\n'.join(links_alt)

with open("week3/wiki_links.txt","w",encoding="utf-8") as f:
     f.write(links_alt_txt)
for img in image_urls:
     print(img)