import requests
from bs4 import BeautifulSoup
import json
import time
import random
import re
import os
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urljoin, unquote
import sys
import io
from functools import partial

# ==============================================================================
# ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 只需要修改這裡 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼
# ==============================================================================
# 1. 設定要爬取的目標
#    - TARGET_AREA: 請從下方執行後顯示的「可選區域列表」中，完整複製一個名稱貼上。
#    - TARGET_KANA_ROW: 填寫 'all' 或假名行的代表字母 (如 'a', 'k', 's', 't' 等)。
TARGET_CITY = "tokyo"
TARGET_AREA = "九段下"
TARGET_KANA_ROW = "k"  # <-- 輸入 'k' 會抓取 ka, ki, ku, ke, ko 所有頁面

# 2. (可選) 調整爬蟲效能與行為
MAX_WORKERS = 8
DELAY_RANGE = (0.5, 1.5)
TIMEOUT = 15
# ==============================================================================
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 以上是唯一需要修改的區域 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲
# ==============================================================================


# --- 假名行對照表 (無需修改) ---
KANA_ROW_MAP = {
    'a': ['a', 'i', 'u', 'e', 'o'],
    'k': ['ka', 'ki', 'ku', 'ke', 'ko'],
    's': ['sa', 'shi', 'su', 'se', 'so'],
    't': ['ta', 'chi', 'tsu', 'te', 'to'],
    'n': ['na', 'ni', 'nu', 'ne', 'no'],
    'h': ['ha', 'hi', 'fu', 'he', 'ho'],
    'm': ['ma', 'mi', 'mu', 'me', 'mo'],
    'y': ['ya', 'yu', 'yo'],
    'r': ['ra', 'ri', 'ru', 're', 'ro'],
    'w': ['wa', 'wo'],
    # 濁音/半濁音行
    'g': ['ga', 'gi', 'gu', 'ge', 'go'],
    'z': ['za', 'ji', 'zu', 'ze', 'zo'],
    'd': ['da', 'ji', 'zu', 'de', 'do'], # 這裡的 ji, zu 是常見替代
    'b': ['ba', 'bi', 'bu', 'be', 'bo'],
    'p': ['pa', 'pi', 'pu', 'pe', 'po'],
    'v': ['v'] # 特殊行
}

# --- 專案路徑與全域設定 (無需修改) ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_JSON_DIR = os.path.join(BASE_DIR, "data", "raw")
PHOTO_DIR = os.path.join(BASE_DIR, "data", "photos")

session = requests.Session()
session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
    "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.8,en;q=0.7",
})

# --- 核心爬蟲函式 (無需修改) ---

def fetch_url(url):
    try:
        time.sleep(random.uniform(*DELAY_RANGE))
        response = session.get(url, timeout=TIMEOUT, headers={"Referer": f"https://tabelog.com/sitemap/{TARGET_CITY}/"})
        response.raise_for_status()
        return response
    except requests.exceptions.RequestException as e:
        print(f"  - 請求錯誤 {url}: {e}", file=sys.stderr)
        return None

def get_named_links_from_page(page_url, selector):
    res = fetch_url(page_url)
    if not res: return {}
    soup = BeautifulSoup(res.text, "html.parser")
    # 將區域名稱中的半形空格替換為全形空格，以匹配網站上的顯示
    return {a.get_text(strip=True).replace(" ", "　"): urljoin(page_url, a["href"]) for a in soup.select(selector) if a.get("href")}

def get_available_kana_paths(area_url):
    """從區域頁面獲取所有可用的假名 URL 路徑 (e.g., ['a', 'i', 'ka', 'ki'])"""
    res = fetch_url(area_url)
    if not res: return []
    soup = BeautifulSoup(res.text, "html.parser")
    paths = []
    for a_tag in soup.select("ul.sitemap-50on__list li a"):
        href = a_tag.get("href", "").strip('/')
        if href:
            path_part = os.path.basename(href)
            if path_part:
                paths.append(path_part)
    return paths

def get_photo_urls(restaurant_url):
    photo_urls = {"exterior": [], "interior": []}
    photo_category_pages = {"interior": urljoin(restaurant_url, "dtlphotolst/3/smp2/"), "exterior": urljoin(restaurant_url, "dtlphotolst/4/smp2/")}
    for category, page_url in photo_category_pages.items():
        current_url = page_url
        while current_url:
            res = fetch_url(current_url)
            if not res or res.status_code == 404: break
            soup = BeautifulSoup(res.text, "html.parser")
            img_tags = soup.select('div.rstdtl-thumb-list__img-wrap img')
            for img in img_tags:
                img_url = img.get("src")
                if img_url:
                    high_res_url = re.sub(r'/150x150_square_/', '/1200x1200_square_/', img_url)
                    photo_urls[category].append(high_res_url)
            next_page_tag = soup.select_one("a.c-pagination__arrow--next")
            current_url = urljoin(current_url, next_page_tag["href"]) if next_page_tag else None
    return photo_urls

def get_restaurant_basic(url):
    res = fetch_url(url)
    if not res: return None
    soup = BeautifulSoup(res.text, "html.parser")
    rating_tag = soup.select_one('span.rdheader-rating__score-val-dtl')
    genre_tag = soup.select_one("th:-soup-contains('ジャンル') + td")
    address_tag = soup.select_one('p.rstinfo-table__address')
    hours_tag = soup.select_one("th:-soup-contains('営業時間') + td")
    
    # ===== 修正最近車站的爬取邏輯 =====
    station_info = None
    
    # 方法1: 從header區域的最寄り駅獲取
    station_dt = soup.find('dt', string=lambda text: text and '最寄り駅' in text)
    if station_dt:
        station_dd = station_dt.find_next_sibling('dd')
        if station_dd:
            station_span = station_dd.select_one('span.linktree__parent-target-text')
            if station_span:
                station_info = station_span.get_text(strip=True)
    
    # 備選方法: 如果上面沒找到，嘗試表格中的交通手段
    if not station_info:
        station_tag_alt = soup.select_one("th:-soup-contains('交通手段') + td")
        if station_tag_alt:
            station_text = station_tag_alt.get_text(strip=True)
            station_match = re.search(r'([^、\n]*駅)', station_text)
            if station_match:
                station_info = station_match.group(1).strip()
    # =======================================
    
    budget_dinner_tag = soup.select_one(".rstinfo-table__budget-item:has(.c-rating-v3__time--dinner) em")
    budget_lunch_tag = soup.select_one(".rstinfo-table__budget-item:has(.c-rating-v3__time--lunch) em")
    budget = {}
    if budget_dinner_tag: budget['dinner'] = budget_dinner_tag.get_text(strip=True)
    if budget_lunch_tag: budget['lunch'] = budget_lunch_tag.get_text(strip=True)
    
    return {
        "rating": rating_tag.get_text(strip=True) if rating_tag else None,
        "genre": genre_tag.get_text(strip=True).replace("\n", " ") if genre_tag else None,
        "address": address_tag.get_text(strip=True).replace("大きな地図を見る", "").strip() if address_tag else None,
        "station_info": station_info,  # 使用修正後的車站資訊
        "hours": hours_tag.get_text(strip=True) if hours_tag else None,
        "budget": budget if budget else None,
    }

def get_all_review_urls(restaurant_url):
    review_list_url = urljoin(restaurant_url, "dtlrvwlst/")
    all_review_urls = []
    current_url = review_list_url
    while current_url:
        res = fetch_url(current_url)
        if not res: break
        soup = BeautifulSoup(res.text, "html.parser")
        review_links = soup.select("a.js-link-bookmark-detail[data-detail-url]")
        if not review_links: break
        for link in review_links:
            detail_url_path = link.get('data-detail-url')
            if detail_url_path:
                all_review_urls.append(urljoin("https://tabelog.com", detail_url_path))
        next_page_tag = soup.select_one("a.c-pagination__arrow--next")
        current_url = urljoin(current_url, next_page_tag["href"]) if next_page_tag else None
    return all_review_urls

def get_review_detail(review_url):
    res = fetch_url(review_url)
    if not res: return None
    soup = BeautifulSoup(res.text, "html.parser")
    title_tag = soup.select_one("p.rvw-item__title strong")
    comment_tag = soup.select_one("div.rvw-item__rvw-comment p")
    title_text = title_tag.get_text(strip=True) if title_tag else None
    comment_text = comment_tag.get_text(strip=True) if comment_tag else None
    if not title_text and not comment_text:
        return None
    user_rating_tag = soup.select_one("b.rvw-item__ratings--val")
    detail_ratings = {}
    for li in soup.select("ul.c-rating-detail li"):
        spans = li.find_all(recursive=False)
        if len(spans) == 2:
            label_text = spans[0].get_text(strip=True)
            score_text = spans[1].get_text(strip=True)
            detail_ratings[label_text] = score_text
        elif len(spans) == 1 and li.strong:
            label_text = li.strong.previous_sibling.strip()
            score_text = li.strong.get_text(strip=True)
            if label_text:
                 detail_ratings[label_text] = score_text
    return {
        "title": title_text,
        "comment": comment_text,
        "user_rating": user_rating_tag.get_text(strip=True) if user_rating_tag else None,
        "detail_ratings": detail_ratings if detail_ratings else None,
        "url": review_url
    }

def download_photo(url, filepath):
    if os.path.exists(filepath): return True
    try:
        res = session.get(url, stream=True, timeout=TIMEOUT)
        res.raise_for_status()
        with open(filepath, 'wb') as f: f.write(res.content)
        return True
    except requests.exceptions.RequestException: return False

def process_restaurant(restaurant_data, photo_dir):
    name, url = restaurant_data
    try:
        tabelog_id = re.search(r'/(\d+)/?$', url).group(1)
    except (AttributeError, IndexError): return None
    print(f"🔄 處理中: {name} (ID: {tabelog_id})")
    detail = get_restaurant_basic(url)
    if not detail:
        print(f"  ❌ 無法取得基本資料: {name}", file=sys.stderr)
        return None
    
    review_urls = get_all_review_urls(url)
    photo_urls = get_photo_urls(url)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        review_futures = [executor.submit(get_review_detail, r_url) for r_url in review_urls]
        photo_futures, local_photo_filenames = [], {k: [] for k in photo_urls}
        for category, urls in photo_urls.items():
            for i, p_url in enumerate(urls):
                ext = os.path.splitext(unquote(p_url.split('?')[0]))[1] or '.jpg'
                filename = f"{tabelog_id}_{category}_{i+1:02d}{ext}"
                filepath = os.path.join(photo_dir, filename)
                photo_futures.append(executor.submit(download_photo, p_url, filepath))
                local_photo_filenames[category].append(filename)
        reviews = [f.result() for f in review_futures if f.result()]
        [f.result() for f in photo_futures]
    result = {"tabelog_id": tabelog_id, "name": name, "url": url, **detail, "photos": local_photo_filenames, "reviews": reviews}
    print(f"✅ 完成: {name} (已抓取評論: {len(reviews)}, 照片: {sum(len(v) for v in photo_urls.values())})")
    return result

def scrape_restaurant_list_page(page_url, photo_dir):
    restaurants = []
    current_url = page_url
    process_func = partial(process_restaurant, photo_dir=photo_dir)
    while current_url:
        print(f"\n--- 正在爬取列表頁面: {current_url} ---")
        res = fetch_url(current_url)
        if not res: break
        soup = BeautifulSoup(res.text, "html.parser")
        restaurant_links = soup.select("a.sitemap-50dtl__name")
        if not restaurant_links: break
        restaurant_data = [(a.get_text(strip=True), urljoin(current_url, a["href"])) for a in restaurant_links]
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(process_func, restaurant_data))
            restaurants.extend([r for r in results if r])
        next_page_tag = soup.select_one("a.c-pagination__arrow--next")
        current_url = urljoin(current_url, next_page_tag["href"]) if next_page_tag else None
    return restaurants

def run_pre_flight_check():
    print("--------------------------------------------------")
    print("🚀 Tabelog 爬蟲啟動前檢查...")
    city_url = f"https://tabelog.com/sitemap/{TARGET_CITY}/"
    area_links = get_named_links_from_page(city_url, "ul.area-content__list a.area-content__item-target")
    if not area_links:
        print(f"❌ 錯誤：無法從 {city_url} 獲取任何主要區域列表。")
        sys.exit(1)
    
    sub_area_links = get_named_links_from_page(city_url, "ul.area-list__item-sub a")
    area_links.update(sub_area_links)
    
    print(f"\n🏙️ 在 {TARGET_CITY.upper()} 中找到以下可選區域列表：")
    for name in sorted(area_links.keys()): print(f"  - {name}")
    if TARGET_AREA not in area_links:
        print("\n" + "="*50)
        print(f"❌ 設定錯誤：您設定的目標區域 '{TARGET_AREA}' 不在可選列表中。")
        print("👉 請從上面的列表中，完整複製一個有效的區域名稱到檔案頂部的 TARGET_AREA。")
        print("="*50)
        sys.exit(1)
    print("\n✅ 區域設定正確！")
    return area_links[TARGET_AREA]

def main():
    if sys.stdout.encoding != 'utf-8':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    target_area_url = run_pre_flight_check()

    print("\n--------------------------------------------------")
    print(f"🎯 準備爬取目標:")
    print(f"   - 城市: {TARGET_CITY}")
    print(f"   - 區域: {TARGET_AREA}")
    print(f"   - 假名行: {TARGET_KANA_ROW}")
    print("--------------------------------------------------\n")
    time.sleep(2)

    os.makedirs(RAW_JSON_DIR, exist_ok=True)
    os.makedirs(PHOTO_DIR, exist_ok=True)
    
    urls_to_scrape = []
    input_kana_row = TARGET_KANA_ROW.lower()

    # <<<--- 全新、符合您期望的 URL 生成邏輯 ---<<<
    available_paths = get_available_kana_paths(target_area_url)
    if not available_paths:
        print(f"❌ 錯誤: 無法從 '{TARGET_AREA}' 頁面獲取任何可用的假名行連結。")
        return

    if input_kana_row == 'all':
        print(f"🔍 模式 'all': 準備爬取所有 {len(available_paths)} 個可用的假名行。")
        urls_to_scrape = [urljoin(target_area_url, f"{path}/") for path in available_paths]
    
    elif input_kana_row in KANA_ROW_MAP:
        # 從對照表獲取該行的所有羅馬拼音
        romaji_in_row = KANA_ROW_MAP[input_kana_row]
        
        # 篩選出在該頁面實際存在的路徑
        paths_to_scrape = [r for r in romaji_in_row if r in available_paths]
        
        if not paths_to_scrape:
            print(f"❌ 資訊: 在 '{TARGET_AREA}' 區域中，沒有找到任何屬於 '{input_kana_row}' 行的餐廳分頁。")
            print(f"   (可用的分頁有: {', '.join(available_paths)})")
            return
            
        print(f"✅ 找到 '{input_kana_row}' 行中可用的分頁: {', '.join(paths_to_scrape)}")
        urls_to_scrape = [urljoin(target_area_url, f"{path}/") for path in paths_to_scrape]

    else:
        print(f"❌ 設定錯誤: TARGET_KANA_ROW '{TARGET_KANA_ROW}' 不是 'all' 或一個有效的行代表字母。")
        print(f"👉 可用的行代表字母有: {', '.join(KANA_ROW_MAP.keys())}")
        return

    if not urls_to_scrape:
        print("ℹ️ 沒有找到任何需要爬取的 URL，程式結束。")
        return
    # <<<--------------------------------------------------<<<

    all_restaurants = []
    for kana_url in urls_to_scrape:
        restaurants = scrape_restaurant_list_page(kana_url, PHOTO_DIR)
        all_restaurants.extend(restaurants)
        if restaurants:
            safe_area = re.sub(r'[\\/:*?"<>|]', '-', TARGET_AREA)
            row_code = os.path.basename(kana_url.strip('/'))
            filename = f"{TARGET_CITY}_{safe_area}_{row_code}.json"
            filepath = os.path.join(RAW_JSON_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(restaurants, f, ensure_ascii=False, indent=2)
            print(f"\n💾 數據已儲存至: {filepath}")

    print("\n\n" + "="*50)
    print(f"🏁 全部完成！本次共爬取了 {len(all_restaurants)} 間餐廳。")
    print(f"   - JSON 檔案位於: {RAW_JSON_DIR}")
    print(f"   - 照片檔案位於: {PHOTO_DIR}")
    print("="*50)

if __name__ == "__main__":
    main()