# 🚗 VinFast AI Advisor — Chatbot Tư Vấn Xe VinFast

> Nhóm Z-C401 · Practical AI Hackathon

## Thành viên

| Họ và tên | MSSV | Phân công |
|-----------|------|-----------|
| Lê Văn Tùng | 2A202600111 | Search Tools (YouTube, Reddit, Showroom) |
| Nguyễn Đức Sĩ | 2A202600152 | Xây dựng Database & RAG |
| Lê Thành Thưởng | 2A202600106 | Xây dựng Database & RAG |
| Đinh Thái Tuấn | 2A202600360 | UI (Chainlit) & Eval Metrics |

## Giới thiệu

Agent Chatbot hỗ trợ khách hàng tư vấn mua xe VinFast — tra cứu thông số, lọc xe theo ngân sách, tìm showroom, tổng hợp review từ YouTube/Reddit. Mọi câu trả lời đều kèm nguồn trích dẫn.

## Tech Stack

- **LLM**: Qwen 3.5-Flash (qua DashScope API)
- **Framework**: LangGraph + LangChain
- **Vector DB**: ChromaDB (RAG)
- **Search**: Brave Search API (YouTube, Reddit, Showroom)
- **UI**: Chainlit
- **Embeddings**: OpenAI

## Cài đặt

### 1. Clone repo

```bash
git clone <repo-url>
cd ban-Z-C401
```

### 2. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

Hoặc nếu dùng `uv`:

```bash
uv sync
```

### 3. Cấu hình API keys

Tạo file `.env` ở thư mục gốc:

```env
QWEN_API_KEY=your_qwen_api_key
OPENAI_API_KEY=your_openai_api_key
BRAVE_API_KEY=your_brave_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

### 4. Khởi tạo Vector Database (chạy 1 lần)

```bash
python scripts/init_db.py
```

## Chạy ứng dụng

### Chainlit UI (khuyến nghị)

```bash
chainlit run app.py -w
```

Truy cập **http://localhost:8000** trên trình duyệt.

### CLI (test nhanh)

```bash
cd agent
python agent.py
```

## Cấu trúc thư mục

```
├── app.py                  # Chainlit UI entry point
├── agent/
│   └── agent.py            # LangGraph agent + system prompt
├── tools/
│   ├── RAG_tools.py        # RAG search, filter by price, full info
│   └── search_tools.py     # YouTube, Reddit, Showroom search
├── scripts/
│   └── init_db.py          # Khởi tạo ChromaDB
├── data/
│   └── raw/vinfast_clean/  # Dữ liệu xe đã xử lý (16 dòng xe)
├── chroma_db/              # Vector database
├── .chainlit/config.toml   # Cấu hình Chainlit UI
├── chainlit.md             # Welcome screen
└── pyproject.toml          # Dependencies
```

## Tools

| Tool | Mô tả |
|------|-------|
| `tool_rag_search_specific` | Tra cứu chi tiết kỹ thuật (động cơ, pin, nội thất...) |
| `tool_filter_car_by_price` | Lọc xe theo ngân sách |
| `tool_get_full_info` | Lấy toàn bộ thông tin tổng quan 1 dòng xe |
| `tool_search_youtube_reviews` | Tìm video review trên YouTube |
| `tool_search_reddit_comments` | Tìm bình luận thực tế từ Reddit |
| `tool_search_vinfast_showrooms` | Tìm showroom/đại lý VinFast |