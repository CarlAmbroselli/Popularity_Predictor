# Install the Python Requests library:
# `pip install requests`

import requests
import json
import nltk.data

class Translator:
  tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

  def text_to_jobs(text):
    return [{
      "kind": "default",
      "raw_en_sentence": sentence
    } for sentence in Translator.tokenizer.tokenize(text)]

  def translate(text):
    # https://www.deepl.com/jsonrpc
    # POST https://www.deepl.com/jsonrpc

    try:
      response = requests.post(
        url="https://www.deepl.com/jsonrpc",
        headers={
          "Accept-Encoding": "gzip, deflate, br",
          "Origin": "https://www.deepl.com",
          "Accept": "*/*",
          "Content-Type": "text/plain",
          "Referer": "https://www.deepl.com/translator",
          "Cookie": "LMTBID=\"de5d72dd-7ca2-4c8f-91d8-f17185aebc0c\"",
          "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_12_6) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/61.0.3163.100 Safari/537.36",
          "Authority": "www.deepl.com",
          "Accept-Language": "de,en-US;q=0.8,en;q=0.6",
          "X-Requested-With": "XMLHttpRequest",
        },
        data=json.dumps({
          "jsonrpc": "2.0",
          "method": "LMT_handle_jobs",
          "id": 2,
          "params": {
            "lang": {
              "source_lang_computed": "EN",
              "user_preferred_langs": [
                "DE",
                "EN"
              ],
              "target_lang": "DE"
            },
            "priority": -1,
            "jobs": Translator.text_to_jobs(text)
          }
        })
      )
      return " ".join([x['beams'][0]['postprocessed_sentence'] for x in response.json()['result']['translations']])
    except requests.exceptions.RequestException:
      print('HTTP Request failed')
      return ('ERROR DURING TRANSLATION')
    except AttributeError:
      return ('ERROR DURING TRANSLATION')
    except:
      return ('ERROR DURING TRANSLATION')
