# SPEC — AI Product Hackathon

**Nhóm:** Bàn Z-C401  
**Track:** VinFast  
**Problem statement (1 câu):** Người mua xe VinFast hiện phải tự đọc nhiều review rời rạc, mất thời gian và khó kiểm chứng; chatbot AI giúp tổng hợp thông tin theo nhu cầu cá nhân, có dẫn nguồn rõ ràng, để rút ngắn thời gian ra quyết định.

---

## 1. AI Product Canvas

|   | Value | Trust | Feasibility |
|---|-------|-------|-------------|
| **Câu hỏi** | User nào? Pain gì? AI giải gì? | Khi AI sai thì sao? User sửa bằng cách nào? | Cost/latency bao nhiêu? Risk chính? |
| **Trả lời** | **Người dùng mục tiêu:** người đang cân nhắc mua xe VinFast lần đầu hoặc đổi xe.  <br>**Nỗi đau:** mất 2-5 giờ để đọc nhiều nguồn, thông tin mâu thuẫn, khó so sánh theo nhu cầu thực tế (đi phố, gia đình, ngân sách).  <br>**Giá trị AI:** tóm tắt ưu/nhược điểm theo từng mẫu xe (VF3, VF5, VF8...), đề xuất 1-2 lựa chọn phù hợp và nêu lý do có thể kiểm chứng. | **Rủi ro chính:** AI nêu sai thông số, sai giá, hoặc trích sai nguồn.  <br>**Cách giảm rủi ro:** chỉ trả lời dựa trên dữ liệu đã lập chỉ mục; bắt buộc hiển thị nguồn cho thông tin quan trọng (giá, pin, an toàn).  <br>**Khi AI sai:** người dùng bấm “Sai/Thiếu nguồn”, chọn loại lỗi và gửi chỉnh sửa; hệ thống ghi correction log để cải thiện dữ liệu và prompt theo chu kỳ. | **Kỹ thuật:** RAG (truy xuất tài liệu rồi mới sinh câu trả lời), có bước xếp hạng lại kết quả truy xuất.  <br>**Mục tiêu tốc độ:** P95 < 4 giây (95% câu trả lời hoàn tất dưới 4 giây).  <br>**Mục tiêu chi phí:** <= $0.02/câu hỏi.  <br>**Rủi ro vận hành:** dữ liệu cũ, thiên lệch nguồn, thiếu dữ liệu cho phiên bản xe mới. |

**Automation hay augmentation?** Augmentation  
Justify: AI hỗ trợ tư vấn ban đầu; quyết định mua, lái thử và chốt sale vẫn do khách hàng và tư vấn viên showroom.

**Learning signal:**

1. Dữ liệu chỉnh sửa của người dùng đi vào đâu? Vào correction log gồm: loại lỗi, câu hỏi, câu trả lời, nguồn, mẫu xe, thời điểm, và hành động sửa.
2. Sản phẩm đo gì để biết tốt lên hay tệ đi? Tỷ lệ “Hữu ích”, tỷ lệ “Sai/Thiếu nguồn”, tỷ lệ người dùng hỏi sâu (>= 3 câu/phiên), tỷ lệ để lại thông tin liên hệ hợp lệ.
3. Dữ liệu thuộc loại nào? **Loại chính là Domain-specific**. Ngoài ra có Human-judgment (feedback/chỉnh sửa), Real-time một phần (giá/khuyến mãi), và User-specific mức nhẹ (ngữ cảnh phiên chat).  
   Có marginal value không? Có. Dữ liệu correction nội bộ và dữ liệu review tiếng Việt theo ngữ cảnh sử dụng thực tế tạo lợi thế tích lũy theo thời gian.

---

## 2. User Stories — 4 paths

### Feature: Tư vấn chọn xe theo nhu cầu

**Trigger:** Người dùng nhập nhu cầu (ngân sách, số chỗ, quãng đường/ngày, ưu tiên tiết kiệm hay tiện nghi) -> AI đề xuất 1-2 mẫu xe phù hợp, kèm lý do và nguồn.

| Path | Câu hỏi thiết kế | Mô tả |
|------|-------------------|-------|
| Happy — AI đúng, tự tin | Người dùng thấy gì? Kết thúc ra sao? | AI gợi ý “VF5 phù hợp đi phố 30-50 km/ngày trong ngân sách X”, nêu 3 lý do và trích nguồn; người dùng bấm “Hữu ích”, sau đó xem bảng so sánh chi tiết. |
| Low-confidence — AI không chắc | Hệ thống báo “không chắc” thế nào? | Khi dữ liệu thiếu/mâu thuẫn, AI hiển thị “Độ chắc chắn thấp”, đưa tối đa 2 phương án và hỏi thêm câu làm rõ nhu cầu. |
| Failure — AI sai | Người dùng phát hiện sai bằng cách nào? | AI đề xuất xe vượt ngân sách hoặc lệch nhu cầu; người dùng bấm “Sai”; hệ thống xin lỗi, hỏi lại điều kiện bắt buộc, sau đó truy xuất lại và trả lời mới. |
| Correction — người dùng sửa | Người dùng sửa bằng cách nào? Dữ liệu đi đâu? | Người dùng chỉnh ngân sách/tiêu chí và bấm “Lưu phản hồi”; dữ liệu vào correction log để cải thiện bước hiểu ý định và xếp hạng kết quả. |

### Feature: Hỏi đáp thông số và review có dẫn nguồn

**Trigger:** Người dùng hỏi về thông số, giá, hoặc trải nghiệm thực tế của một mẫu xe cụ thể -> AI trả lời ngắn gọn, có dẫn nguồn.

| Path | Câu hỏi thiết kế | Mô tả |
|------|-------------------|-------|
| Happy — AI đúng, tự tin | Người dùng thấy gì? Kết thúc ra sao? | AI trả lời đúng trọng tâm, kèm 2-3 nguồn liên quan; người dùng tiếp tục hỏi sâu hoặc chọn nhận tư vấn từ showroom. |
| Low-confidence — AI không chắc | Hệ thống báo “không chắc” thế nào? | Nếu dữ liệu cũ hoặc thiếu nguồn tin cậy, AI không khẳng định tuyệt đối; hiển thị cảnh báo “Cần xác minh tại showroom”. |
| Failure — AI sai | Người dùng phát hiện sai bằng cách nào? | AI nêu sai giá theo phiên bản; người dùng bấm “Thiếu chính xác” và chọn “Sai giá”; hệ thống buộc truy xuất lại từ nguồn giá mới nhất trước khi trả lời lại. |
| Correction — người dùng sửa | Người dùng sửa bằng cách nào? Dữ liệu đi đâu? | Người dùng gửi nguồn/giá bổ sung; dữ liệu vào hàng chờ kiểm duyệt trước khi cập nhật kho tri thức để tránh nhiễu dữ liệu. |

---

## 3. Eval metrics + threshold

**Optimize precision hay recall?** Precision  
Tại sao? Bài toán tư vấn mua xe cần độ đúng cao; sai thông tin giá/an toàn làm mất niềm tin ngay từ lần đầu.  
Nếu chọn sai hướng thì sao? Nếu thiên về recall quá mức, chatbot có thể trả lời nhiều nhưng kém chắc chắn, làm tăng lỗi và giảm tỷ lệ để lại thông tin liên hệ.

| Metric | Threshold | Red flag (dừng khi) |
|--------|-----------|---------------------|
| Độ chính xác thông tin có dẫn nguồn (factual precision) | >= 90% | < 80% trong 1 tuần |
| Tỷ lệ bịa thông tin ở nhóm giá/an toàn | = 0% | > 1 lỗi xác nhận sai/ngày |
| Tỷ lệ câu trả lời có dẫn nguồn khi có số liệu | >= 95% | < 85% trong 3 ngày liên tiếp |
| Tỷ lệ người dùng bấm “Hữu ích” | >= 70% | < 50% trong 2 tuần |
| P95 latency (95% phản hồi dưới ngưỡng) | < 4 giây | > 6 giây trong giờ cao điểm, 3 ngày liên tiếp |
| Tỷ lệ để lại thông tin liên hệ hợp lệ | >= 5% | < 2% trong 4 tuần |

---

## 4. Top 3 failure modes

| # | Trigger | Hậu quả | Mitigation |
|---|---------|---------|------------|
| 1 | Câu hỏi mơ hồ, có từ lóng/châm biếm, hệ thống hiểu sai ý định | Tư vấn sai nhu cầu nhưng trả lời rất tự tin | Bắt buộc hỏi làm rõ khi điểm tự tin hiểu ý định thấp |
| 2 | Dữ liệu giá/khuyến mãi cũ hoặc lẫn phiên bản xe | Người dùng nhận thông tin giá sai, giảm niềm tin vào hệ thống | Đặt thời hạn hiệu lực dữ liệu giá; nếu quá hạn thì hiển thị “Cần xác minh tại showroom” |
| 3 | Truy xuất nhầm nguồn (khác hãng hoặc nhầm mẫu xe tên gần giống) | AI trích sai nguồn và đề xuất lệch | Lọc cứng theo hãng/mẫu xe trước khi sinh câu trả lời; kiểm tra chéo nội dung trả lời với metadata nguồn |

**Failure nguy hiểm nhất:** AI trả lời nghe hợp lý nhưng sai thực tế, và người dùng không phát hiện ngay.

---

## 5. ROI 3 kịch bản

**Định nghĩa chung:** “Thông tin liên hệ hợp lệ” là bản ghi có ít nhất số điện thoại hoặc email + nhu cầu mua xe cơ bản, đủ để đội tư vấn liên hệ lại.

|   | Conservative | Realistic | Optimistic |
|---|-------------|-----------|------------|
| **Assumption** | 120 người dùng/ngày, 55% thấy hữu ích, 3% để lại thông tin liên hệ hợp lệ | 400 người dùng/ngày, 70% thấy hữu ích, 6% để lại thông tin liên hệ hợp lệ | 1200 người dùng/ngày, 80% thấy hữu ích, 9% để lại thông tin liên hệ hợp lệ |
| **Cost** | $45/ngày (LLM + embedding + hạ tầng) | $140/ngày | $360/ngày |
| **Benefit** | Tiết kiệm 3 giờ tư vấn viên/ngày + 3.6 khách để lại thông tin liên hệ hợp lệ/ngày | Tiết kiệm 10 giờ/ngày + 24 khách để lại thông tin liên hệ hợp lệ/ngày | Tiết kiệm 28 giờ/ngày + 108 khách để lại thông tin liên hệ hợp lệ/ngày |
| **Net** | Dương nhẹ nếu mỗi khách để lại thông tin hợp lệ tạo giá trị >= $15 | Dương rõ rệt, có thể hoàn vốn vận hành theo ngày | Dương mạnh, phù hợp mở rộng đa kênh (web, app, kiosk showroom) |

**Kill criteria:** Dừng hoặc đổi hướng nếu trong 8 tuần liên tiếp có một trong các điều kiện: (1) tỷ lệ “Hữu ích” < 50%, (2) lỗi sai giá/an toàn chưa đưa về 0, hoặc (3) chi phí vận hành lớn hơn lợi ích định lượng trong 2 tháng liên tục.

---

## 6. Mini AI spec (1 trang)

Chatbot tập trung vào giai đoạn người dùng tìm hiểu trước khi mua xe VinFast. Mục tiêu là giúp người dùng giảm thời gian tổng hợp thông tin, có góc nhìn ngắn gọn nhưng đáng tin, và đi đến bước tiếp theo như so sánh xe, đăng ký lái thử, hoặc để lại thông tin liên hệ để showroom tư vấn.

Về kỹ thuật, hệ thống dùng RAG: truy xuất dữ liệu liên quan trước, rồi mới sinh câu trả lời. Với thông tin nhạy cảm như giá và an toàn, chatbot chỉ trả lời khi có nguồn rõ ràng. Nếu dữ liệu chưa đủ chắc chắn, chatbot phải thông báo “độ chắc chắn thấp” và hướng người dùng sang bước xác minh tại showroom.

Sản phẩm ưu tiên precision hơn recall vì rủi ro chính là mất niềm tin khi trả lời sai. Các chỉ số vận hành chính gồm: factual precision có dẫn nguồn, tỷ lệ bịa thông tin bằng 0 cho nhóm giá/an toàn, P95 latency dưới 4 giây, tỷ lệ “Hữu ích”, và tỷ lệ để lại thông tin liên hệ hợp lệ.

Vòng lặp cải thiện được thiết kế ngay từ đầu: mọi phản hồi “Sai/Thiếu nguồn” và chỉnh sửa của người dùng được ghi lại trong correction log, sau đó đưa vào quy trình rà soát và cập nhật định kỳ. Nhờ vậy, hệ thống cải thiện theo dữ liệu thực tế thay vì chỉ dựa vào prompt tĩnh.
