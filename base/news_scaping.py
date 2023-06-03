import pandas as pd
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from .text_cleaner import TextCleaner

# API_key = '3ef72ffe3675480d80cfd3e98937772d'
API_key = "1e62b5142a01450aa6fc8b5a8195d552"



def remove_tags(html):
    soup = BeautifulSoup(html, "html.parser")
  
    for data in soup(['style', 'script']):
        data.decompose()
        
    return ' '.join(soup.stripped_strings) 

def get_interested_content(data_dict, idx):
  content = data_dict['articles'][idx]['content']
  if content is None:
    return 'None'

  try:
    html_content = requests.get(data_dict['articles'][idx]['url']).text
    cleaned_html_content = remove_tags(html_content)

    start_real_content_stmt = content.split()[0]
  except:
    return "None"

  #to avoid empty start word indicator
  if start_real_content_stmt == '…':
    return 'None'

  start_index = cleaned_html_content.find(start_real_content_stmt)
  if start_index == -1:
    return 'None'

  #number of interested content characters indicator
  if content.find('[') == -1:
    return 'None'
  limit = re.findall('[0-9]', content[content.find('['):])
  end_index = int(''.join(limit))+200

  real_content = cleaned_html_content[start_index:end_index]

  return real_content



def build_data():
  data = []
  country_code_list = []
  countries_chuncks = 'aearataubebgbrcachcncocuczdeegfrgbgrhkhuidieilinitjpkrltlvmamxmyngnlnonzphplptrorsrusasesgsiskthtrtwuausveza'

  for i in range(0, len(countries_chuncks), 2):
    country_code_list.append(countries_chuncks[i:i+2])

    categories = ['general', 'health', 'sports', 'science', 'business', 'technology', 'entertainment']
    for category in categories:
      for country_code in country_code_list:
        url = f'https://newsapi.org/v2/top-headlines?language=en&category={category}&country={country_code}&apiKey={API_key}'
        response = requests.get(url)
        data_dict = response.json()

        if data_dict['status'] == 'ok':
          for idx in range(len(data_dict['articles'])):
            title = data_dict['articles'][idx]['title']

            interested_content = get_interested_content(data_dict, idx)
            print(len(interested_content))

            url = data_dict['articles'][idx]['url']
            source = urlparse(url).netloc
            category = category
            country_code = country_code
            date = data_dict['articles'][idx]['publishedAt']

            data.append([title, interested_content, url, source, category, country_code, date])
        else:
          print(data_dict['message'])

  tc = TextCleaner()
  data['cleaned_interested_content'] = data['interested_content'].apply(lambda interested_content: tc.text_cleaner(interested_content))
  news_data = pd.DataFrame(data, columns=['title', 'interested_content', 'url', 'source', 'category', 'country_code', 'date'])   
  news_data = news_data[news_data["interested_content"].str.contains("None") == False]
  news_data = news_data.dropna()
  
  news_data.to_csv('news_data.csv', index=False)
  return news_data