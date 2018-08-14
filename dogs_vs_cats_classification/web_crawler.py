from bs4 import BeautifulSoup
import requests
import os

os.makedirs('./img/birds1/', exist_ok=True)
for i in range(41,81):

    URL = 'https://pixabay.com/en/photos/bird/?&pagi=' + str(i)
    print(URL)
    html  =  requests.get(URL).text
    soup = BeautifulSoup(html, )
    img_ul = soup.find_all('div', {"class": "item"})




    for div in img_ul:
        imgs = div.find_all('img')
        for img in imgs:
            url = img['src']
            print(url)
            if url == '/static/img/blank.gif': continue
            r = requests.get(url,stream=True)

            # image_name = url.split('/')[-1].split('?')[0]  ### https://www.bluecross.org.uk/sites/default/files/styles/thumbnail_pet/public/pets/185134/448685.jpg?itok=VtN8R4XY
                                                            ### 取最后一个／和？之间的内容
            image_name = url.split('/')[-1]

            with open('./img/birds1/%s' % image_name, 'wb') as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
            print('Saved %s' % image_name)
