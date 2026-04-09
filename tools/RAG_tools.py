import os
import json
import glob
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__)) 
DATA_DIR = os.path.join(BASE_DIR, "data", "raw", "vinfast_clean")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# --- CHỈ KẾT NỐI ĐẾN DB ĐÃ TỒN TẠI ---
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# --- ĐỊNH NGHĨA TOOLS ---

@tool
def tool_filter_car_by_price(max_price: int) -> str:
    """
    CHỈ SỬ DỤNG khi người dùng hỏi về việc mua xe dựa trên ngân sách hoặc số tiền họ có.
    Ví dụ: "Tôi có 300 triệu mua được xe gì?", "Ngân sách 600 triệu".
    Tham số 'max_price': Số tiền tối đa người dùng có (đơn vị VNĐ). Ví dụ: 300 triệu thì truyền vào 300000000.
    """
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    affordable_cars = []
    
    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            car_data = json.load(f)[0]
            price_str = car_data.get("price", {}).get("listed_price", "0")
            price_int = int(''.join(filter(str.isdigit, price_str))) if price_str else 0
            if "triệu" in price_str.lower(): price_int *= 1000000
                
            if 0 < price_int <= max_price + 50000000:
                affordable_cars.append(f"- Xe: {car_data['car_model']} | Giá: {price_str} | Dòng: {car_data['segment']}")
                
    if not affordable_cars:
        return f"Với ngân sách {max_price:,.0f} VNĐ, hiện tại VinFast chưa có dòng xe nào phù hợp. Rẻ nhất là VF3 giá khoảng 302 triệu."
    
    return "Các xe VinFast phù hợp với ngân sách (kể cả cố rướn thêm chút đỉnh) là:\n" + "\n".join(affordable_cars)


@tool
def tool_rag_search_specific(query: str, car_model: str = None) -> str:
    """
    CHỈ SỬ DỤNG khi người dùng hỏi MỘT THÔNG TIN CHI TIẾT (ví dụ: chỉ hỏi động cơ, chỉ hỏi nội thất).
    TUYỆT ĐỐI KHÔNG sử dụng tool này để xin toàn bộ thông tin tổng quan.
    TUYỆT ĐỐI KHÔNG gọi tool này cùng lúc với tool_get_full_info.
    LƯU Ý: Tham số 'query' nên là từ khóa ngắn gọn, trực diện (VD: 'động cơ', 'nội thất', 'pin').
    """
    # BỎ filter cứng. Gộp tên xe vào query (VD: "VF e34 động cơ") để thuật toán tự tìm
    search_query = f"{car_model} {query}" if car_model else query
    
    # TĂNG k=5 để mở rộng vùng lấy dữ liệu, đảm bảo không sót chunk thông số
    results = vector_store.similarity_search(search_query, k=5)
    
    if not results:
        return f"Không tìm thấy thông tin cho: {search_query}"
    
    formatted_results = f"Thông tin trích xuất cho {search_query}:\n"
    for doc in results:
        formatted_results += f"{doc.page_content}\n\n"
        
    return formatted_results


@tool
def tool_get_full_info(car_model: str) -> str:
    """
    CHỈ SỬ DỤNG khi người dùng yêu cầu xin TOÀN BỘ, TẤT CẢ thông tin tổng quan về một chiếc xe.
    TUYỆT ĐỐI KHÔNG dùng tool này nếu user chỉ hỏi 1 khía cạnh (như động cơ, giá bán).
    TUYỆT ĐỐI KHÔNG gọi tool này cùng lúc với tool_rag_search_specific.
    """
    normalized_model = car_model.lower().replace(" ", "_") # "VF 3" -> "vf_3"
    
    matched_files = glob.glob(os.path.join(DATA_DIR, f"vinfast_{normalized_model}*.json"))
    
    if not matched_files:
        return f"Không tìm thấy file dữ liệu tổng quan cho dòng xe: {car_model}"
        
    with open(matched_files[0], 'r', encoding='utf-8') as f:
        car_data = json.load(f)[0]
        
    result = f"=== TỔNG QUAN XE {car_data['car_model']} ===\n"
    result += f"- Phân khúc: {car_data.get('segment', 'Không xác định')}\n"
    result += f"- Giá niêm yết: {car_data.get('price', {}).get('listed_price', 'Đang cập nhật')}\n\n"
    
    for spec in car_data.get("specifications", []):
        result += f"[{spec['category']}]\n"
        for k, v in spec.get("attributes", {}).items():
            result += f"  + {k}: {v}\n"
        result += "\n"
        
    return result