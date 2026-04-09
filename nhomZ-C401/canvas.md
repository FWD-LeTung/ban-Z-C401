# AI Product Canvas — Vietnam Airlines Chatbot NEO

## Canvas

|   | Value | Trust | Feasibility |
|---|-------|-------|-------------|
| **Câu hỏi guide** | User nào? Pain gì? AI giải quyết gì mà cách hiện tại không giải được? | Khi AI sai thì user bị ảnh hưởng thế nào? User biết AI sai bằng cách nào? User sửa bằng cách nào? | Cost bao nhiêu/request? Latency bao lâu? Risk chính là gì? |
| **Trả lời** | **User:** hành khách Vietnam Airlines cần hỗ trợ nhanh về chuyến bay, hành lý, đổi vé, check-in. <br><br> **Pain:** tổng đài thường quá tải, user phải chờ lâu hoặc phải tìm thông tin trong nhiều trang FAQ. <br><br> **AI giải quyết:** chatbot NEO cho phép user hỏi trực tiếp bằng ngôn ngữ tự nhiên và nhận câu trả lời ngay lập tức, giúp giảm thời gian tìm kiếm và giảm tải cho tổng đài. | **Ảnh hưởng khi AI sai:** user có thể nhận thông tin sai về hành lý, đổi vé hoặc check-in, dẫn đến nhầm lẫn hoặc mất thời gian. <br><br> **User biết AI sai:** câu trả lời không khớp với tình huống của họ hoặc khác với thông tin trên website. <br><br> **User sửa:** hỏi lại chatbot, chọn menu FAQ, hoặc chuyển sang tìm thông tin trên website / liên hệ hotline. | **Cost:** nếu dùng LLM API có thể khoảng **$0.001–$0.01/request** (ước lượng). <br><br> **Latency:** khoảng **1–3 giây** cho mỗi câu trả lời. <br><br> **Risk:** AI hiểu sai intent của user, cung cấp thông tin không chính xác, hoặc không xử lý được câu hỏi phức tạp. |

---

## Automation hay augmentation?

☐ Automation — AI làm thay, user không can thiệp  
☑ Augmentation — AI gợi ý, user quyết định cuối cùng  

**Justify:**  
Chatbot NEO chủ yếu cung cấp thông tin và gợi ý, không tự động thực hiện các hành động quan trọng như đổi vé hoặc hoàn tiền. User vẫn phải xác nhận hoặc thực hiện bước cuối cùng. Điều này giúp giảm rủi ro khi AI trả lời sai.

---

## Learning signal

| # | Câu hỏi | Trả lời |
|---|---------|---------|
| 1 | User correction đi vào đâu? | Khi user nhập lại câu hỏi, chọn menu khác hoặc chuyển sang link FAQ, hệ thống có thể ghi lại log để cải thiện intent detection và training data cho chatbot. |
| 2 | Product thu signal gì để biết tốt lên hay tệ đi? | - Tỉ lệ chatbot trả lời thành công <br> - Số lần user phải hỏi lại <br> - Tỉ lệ user chuyển sang hotline hoặc human support <br> - User feedback / rating chatbot |
| 3 | Data thuộc loại nào? | ☑ Domain-specific · ☑ User-specific · ☑ Real-time |

**Giải thích:**

- **Domain-specific:** dữ liệu về quy định hàng không, hành lý, chuyến bay  
- **User-specific:** lịch sử câu hỏi của từng user  
- **Real-time:** thông tin chuyến bay thay đổi liên tục  

---

## Có marginal value không?

Có **marginal value ở mức trung bình**.

Một phần dữ liệu như **FAQ hàng không** đã phổ biến và nhiều chatbot khác cũng có thể truy cập.  

Tuy nhiên, Vietnam Airlines có lợi thế ở:

- dữ liệu nội bộ về chuyến bay  
- hành vi khách hàng  
- log tương tác chatbot  

Những dữ liệu này giúp chatbot **hiểu nhu cầu user tốt hơn theo thời gian**.