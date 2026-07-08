import os
import json
import urllib.parse
import urllib.request
from dotenv import load_dotenv

load_dotenv()

client_id = os.getenv("NAVER_CLIENT_ID")
client_secret = os.getenv("NAVER_CLIENT_SECRET")

def search_stock_news(query, display=10, start=1, sort="sim"):
    if not client_id or not client_secret:
        raise ValueError("Client_ID 또는 Client_SECRET이 맞지 않습니다.")

    enc_text = urllib.parse.quote(query)
    url = (
        "https://openapi.naver.com/v1/search/news.json"
        f"?query={enc_text}"
        f"&display={display}"
        f"&start={start}"
        f"&sort={sort}"
    )

    request = urllib.request.Request(url)
    request.add_header("X-Naver-Client-Id", client_id)
    request.add_header("X-Naver-Client-Secret", client_secret)

    response = urllib.request.urlopen(request)
    response_body = response.read().decode("utf-8")

    return json.loads(response_body)


