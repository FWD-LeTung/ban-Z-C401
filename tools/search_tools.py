import requests
import os
from langchain_core.tools import tool
# Khuyến nghị dùng dotenv để bảo mật API key, không hardcode
from dotenv import load_dotenv
load_dotenv()

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

def _call_brave_api(query: str, count: int = 5) -> dict:
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY
    }
    params = {"q": query, "count": count, "search_lang": "vi"}
    
    try:
        response = requests.get(BRAVE_SEARCH_URL, headers=headers, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}

def search_youtube_reviews(car_model: str) -> str:
    """Tool tìm kiếm video review trên YouTube."""
    query = f"{car_model} đánh giá review site:youtube.com"
    data = _call_brave_api(query, count=3)
    
    if "error" in data:
        return f"Lỗi: {data['error']}"
        
    results = data.get("web", {}).get("results", [])
    if not results:
        return "Không tìm thấy video nào."
    
    # Format rõ ràng nguồn để Agent dễ trích dẫn
    formatted = "Kết quả từ YouTube:\n"
    for item in results:
        formatted += f"- Tiêu đề: {item.get('title')}\n  Mô tả: {item.get('description')}\n  Nguồn (URL): {item.get('url')}\n\n"
    return formatted

def search_reddit_comments(car_model: str, specific_query: str) -> str:
    """
    Tool tìm kiếm thảo luận trên Reddit. 
    specific_query là điều người dùng muốn biết thêm.
    """
    # Kết hợp tên xe và intent của user, ép tìm trên reddit
    query = f"{car_model} {specific_query} site:reddit.com"
    data = _call_brave_api(query, count=5)
    
    if "error" in data:
        return f"Lỗi: {data['error']}"
        
    results = data.get("web", {}).get("results", [])
    if not results:
        return "Không tìm thấy thảo luận phù hợp trên Reddit."
    
    # Format rõ ràng nguồn để Agent trích dẫn (Citation)
    formatted = "Kết quả thảo luận từ Reddit:\n"
    for item in results:
        formatted += f"- Chủ đề: {item.get('title')}\n  Nội dung: {item.get('description')}\n  Nguồn (URL): {item.get('url')}\n\n"
    return formatted

def search_vinfast_showrooms(location: str) -> str:
    """
    Tool tìm kiếm Showroom, đại lý VinFast.
    """
    # Nếu không có location cụ thể, tìm kiếm chung. Nếu có, ghép vào chuỗi tìm kiếm.
    loc_query = f"tại {location}" if location and location.strip() else "gần nhất"
    query = f"Showroom đại lý VinFast {loc_query} địa chỉ số điện thoại hotline giá bán"
    
    data = _call_brave_api(query, count=4)
    
    if "error" in data:
        return f"Lỗi khi tìm kiếm Showroom: {data['error']}"
        
    results = data.get("web", {}).get("results", [])
    if not results:
        return f"Không tìm thấy thông tin Showroom nào {loc_query}."
    
    formatted = f"Kết quả tìm kiếm Showroom VinFast {loc_query}:\n"
    for item in results:
        formatted += f"- Tên trang/Đại lý: {item.get('title')}\n"
        formatted += f"  Thông tin (Địa chỉ/Hotline/Giá): {item.get('description')}\n"
        formatted += f"  Nguồn (URL): {item.get('url')}\n\n"
        
    return formatted

@tool
def tool_fallback_brave_search(car_model: str) -> str:
    """
    CHỈ SỬ DỤNG LÀM BIỆN PHÁP DỰ PHÒNG (FALLBACK) KHI TÌM KIẾM GIÁ XE HOẶC KHUYẾN MÃI.
    Dùng khi khách hàng hỏi giá nhưng hệ thống nội bộ không có, hoặc khách muốn tham khảo giá/khuyến mãi thực tế từ đại lý.
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        return "Lỗi: Chưa cấu hình BRAVE_API_KEY trong file .env"

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key
    }
    
    search_query = f"giá xe {car_model} khuyến mãi site:vnexpress.net/oto-xe-may/v-car/hang-xe/vinfast-33"
    params = {"q": search_query, "count": 3}
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = data.get("web", {}).get("results", [])
        if not results:
            return f"Không tìm thấy thông tin giá cho {car_model} trên website đại lý VinFast Hà Nội 5S."
            
        formatted_results = f"Kết quả tham khảo giá và khuyến mãi {car_model} từ đại lý:\n"
        for item in results:
            formatted_results += f"- Tiêu đề: {item.get('title')}\n  Mô tả: {item.get('description')}\n  Nguồn: {item.get('url')}\n\n"
        return formatted_results
        
    except Exception as e:
        return f"Lỗi khi gọi API tìm kiếm giá đại lý: {str(e)}"