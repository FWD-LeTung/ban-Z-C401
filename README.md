# BanZ_Day05

## Agent Chatbot hỗ trợ mua xe

* LLM: sử dụng api key Qwen 3.6
* Tools:
    * Search_tools: Sử dụng brave api search youtube để trả về link video review, reddit để trả về bình luận của người dùng thực tế về loại xe mà người dùng đang quan tâm. 
    * RAG tool: một tool truy xuất thông tin xe
    * Calculate tool: Một tool tính toán đang trong thời gian thảo luận
* UI: sử dụng gradio hoặc chainlit
* Frame_work: LangGraph