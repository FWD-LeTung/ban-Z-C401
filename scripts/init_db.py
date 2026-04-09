import os
import json
import glob
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

# --- CẤU HÌNH ĐƯỜNG DẪN ---
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data/raw/vinfast_clean")
DB_DIR = os.path.join(BASE_DIR, "chroma_db")

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

def load_and_chunk_json():
    """Đọc các file JSON và chunking theo từng Category kèm Metadata"""
    documents = []
    json_files = glob.glob(os.path.join(DATA_DIR, "*.json"))
    
    if not json_files:
        print("❌ Không tìm thấy file JSON nào trong thư mục data!")
        return []

    for file_path in json_files:
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                car_data = json.load(f)[0]
            except Exception as e:
                print(f"Lỗi đọc file {file_path}: {e}")
                continue
            
            car_model = car_data.get("car_model", "")
            
            # Xử lý giá tiền thành số nguyên để phục vụ Tool Filter
            price_str = car_data.get("price", {}).get("listed_price", "0")
            price_int = int(''.join(filter(str.isdigit, price_str))) if price_str else 0
            if "triệu" in price_str.lower():
                price_int *= 1000000

            # Chunking: Mỗi Category là 1 Document
            for spec in car_data.get("specifications", []):
                category = spec.get("category", "")
                attributes = spec.get("attributes", {})
                
                content = f"Xe: {car_model}\nDanh mục: {category}\nThông số chi tiết:\n"
                for key, value in attributes.items():
                    content += f"- {key}: {value}\n"
                
                metadata = {
                    "car_model": car_model,
                    "category": category,
                    "price_int": price_int,
                    "source": os.path.basename(file_path)
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
    return documents

if __name__ == "__main__":
    print("⏳ Đang đọc dữ liệu JSON và băm nhỏ (chunking)...")
    docs = load_and_chunk_json()
    
    if docs:
        print("⏳ Đang nhúng (embedding) và lưu vào ChromaDB...")
        # Lệnh này sẽ tự động lưu DB vào thư mục persist_directory
        vector_store = Chroma.from_documents(docs, embeddings, persist_directory=DB_DIR)
        vector_store.persist()
        print(f"✅ Hoàn tất! Đã tạo database thành công tại: {DB_DIR}")
        print(f"✅ Tổng số chunks đã nạp: {len(docs)}")
    else:
        print("❌ Quá trình tạo Database thất bại do không có dữ liệu.")