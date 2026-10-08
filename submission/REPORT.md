# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Quốc Đạt  **MSSV**: 2A202602369  **Ngày**: 2026-10-08
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: Tesla T4, 14.6 GB VRAM, fp16

> **Trạng thái: báo cáo smoke test, chưa đủ điều kiện nộp bài.**
> Số liệu NB2–NB5 được chép từ log Colab RUN ALL người học cung cấp, checkout `8c657d9`.
> Run dùng `EVAL_LIMIT=8`: chỉ đánh giá 8 target và 8 regression, thay vì đủ 50 và 15.
> Artefact NB2–NB5 và adapter nằm trên Colab theo log, chưa được tải về repo local để đối chiếu trực tiếp.
> Cần chạy lại NB2 và NB5 với `EVAL_LIMIT=0`, giữ nguyên model, prompt và tập eval, rồi cập nhật báo cáo bằng kết quả đầy đủ.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage, corpus mặc định |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | 1024 theo tier T4; p95 = 98, max = 101, suggested_max_length = 256 |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2 epochs / 30 optimizer steps cho cả bốn cấu hình |

Chọn corpus mặc định để đánh giá khách quan bốn trường intent, urgency, product và sentiment. Model 4B theo tier T4 phù hợp GPU Colab; peak VRAM của LoRA 16-bit đo được là 8.78 GB. T4 dùng fp16 và gradient scaling vì không hỗ trợ bf16. Giữ max_length=1024 theo cấu hình mặc định để các run nhất quán; các mẫu đo ở NB1 đều dưới giới hạn nên không bị cắt, dù 256 đã đủ cho corpus hiện tại.

**Template có giữ khối `<think>` không?** Có: `ok=true`, `open_tag_present=true`, `body_present=true`. NB1 local đã kiểm tra tokenizer thật; log Colab cho cùng số đo. Tổng thời gian NB1–NB5 trong log là 2302 giây, khoảng 38.4 phút. Log ghi 119 unit tests pass; lần kiểm tra masking local trước đó có 23 tests pass. Đây là hai lần kiểm tra khác nhau, không phải một kết quả chạy toàn bộ suite local.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | 0.4149, tức 39 / 94 token trên mẫu minh họa |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn được tính loss trên mẫu minh họa:

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Trên 225 mẫu train, log NB3 ghi 9014 / 20951 token được supervise, khoảng 43.0%. Mask đúng chứng minh phạm vi tính loss; nó không tự chứng minh model sẽ tổng quát tốt sau huấn luyện.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7500 | 0.0000 | 3580.4 |
| (b) base + optimized prompt | 0.6875 | 0.7500 | 1.0000 | 1058.5 |
| (c) LoRA fine-tune | 0.9688 | 0.6667 | 1.0000 | 1508.3 |

**Phạm vi:** 8 target / 8 regression, smoke mode. NB2 đo (a), (b) trước NB3; (c) được đo tại NB5. `optimized_prompt_sha=719e74d3b6232053`. Gatekeeper trong log xác nhận prompt và tập eval không bị sửa.

**(b) có thật sự mạnh hơn (a) không?** Có trên tập smoke: target tăng 0.6875, format tăng từ 0 lên 1, regression giữ nguyên và latency giảm. Không có bằng chứng chỉnh `OPTIMIZED_PROMPT` trong run này; dùng prompt mặc định của repo. Không thể lấy kết quả so với prompt yếu (a) làm bằng chứng duy nhất cho giá trị của fine-tune.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32464896 | 1e-4 | 0.6266 | 0.9688 | 417.9 | 8.78 |
| `attn_only` | q,v | 283 (matched) | 32456704 | 1e-4 | 0.5379 | 0.9375 | 280.6 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32464896 | 1e-5 | 1.5702 | 0.0000 | 414.2 | 8.78 |
| `qlora` | text-linear | 16 | 32464896 | 1e-4 | 0.7058 | 0.8438 | 484.2 | 3.86 |

Cả bốn run có cùng ngân sách 30 optimizer steps. Cột train loss là `training_loss` tổng hợp được ghi vào `final_loss`, không phải loss của riêng bước cuối. Log `correct` có loss bước cuối khoảng 0.02174 nhưng giá trị tổng hợp là 0.6266. Log cũng có `grad_norm=nan` ở một số thời điểm; chưa đủ bằng chứng để kết luận các bước cập nhật tương ứng đều hợp lệ.

**4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về rank so với vị trí gắn adapter?**

`attn_only` thua `correct` trên target smoke: 0.9375 so với 0.9688, dù train loss thấp hơn (0.5379 so với 0.6266). Số tham số lệch 8192, khoảng 0.0252%, nên hai cấu hình đáp ứng điều kiện khớp ngân sách dưới 5%. Thứ tự theo train loss trái với thứ tự theo target. Tăng rank q,v lên 283 chưa bù được lợi ích của vị trí text-linear ở run này; tuy nhiên chênh lệch chỉ tương ứng một trường đúng trên 32 trường của 8 ticket, cần full eval trước khi kết luận rộng hơn.

**4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?**

LR giảm từ 1e-4 xuống 1e-5, giữ placement, rank và ngân sách bước như `correct`. Các mốc log của `wrong_lr` giảm từ 2.163 xuống 2.066, 1.606, 1.326, 1.141 và 1.119; `correct` giảm từ 2.163 xuống 0.02174 ở mốc cuối. Training loss tổng hợp tương ứng là 1.5702 và 0.6266, còn target và format của `wrong_lr` đều bằng 0 ở NB5. Đường loss vẫn giảm nhưng chậm hơn rõ rệt. Nếu chỉ thấy xu hướng giảm và bỏ qua LR, ngân sách bước cùng eval, tôi có thể kết luận nhầm rằng model đã học đủ; ở đây bằng chứng trực tiếp là nó chưa đáp ứng hợp đồng JSON trong tập smoke.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?**

QLoRA giảm peak VRAM từ 8.78 xuống 3.86 GB, tiết kiệm 4.92 GB, khoảng 56.0%. Đổi lại, target giảm 0.1250, train lâu hơn khoảng 15.9% và latency tăng từ 1508.3 lên 1844.4 ms, khoảng 22.3%; format vẫn là 1.0. Số đo smoke ủng hộ việc ưu tiên LoRA 16-bit khi đủ VRAM, nhưng chưa chứng minh QLoRA luôn kém trên mọi dữ liệu. QLoRA vẫn là lựa chọn có thể cân nhắc khi bộ nhớ là giới hạn chính, sau full eval và kiểm tra regression riêng.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`
`target Δ = +0.28125` · `regression Δ = -0.08333` · `valid_trace_rate = 0.0`

Fine-tune tăng target từ 0.6875 lên 0.96875, tương đương 28.125 điểm phần trăm trên tập smoke. Tuy nhiên, regression giảm từ 0.75 xuống 0.6667, vượt mức giảm cho phép 0.02. Vì cổng yêu cầu đồng thời cải thiện target và giữ khả năng tổng quát, kết quả cuối là FAILED. JSON đúng định dạng ở toàn bộ tám target không thể thay thế điều kiện regression. Latency cũng tăng khoảng 42.5% so với baseline prompt tối ưu, nên lợi ích tác vụ có thêm chi phí suy luận. Một hướng thử là thêm 1–5% dữ liệu replay kiến thức tổng quát rồi huấn luyện lại, nhưng đây là đề xuất chưa được thực nghiệm trong log. `valid_trace_rate=0` không đủ để xác nhận reasoning-trace collapse vì chưa có cặp thí nghiệm mask và dữ liệu reasoning tương ứng. Quan trọng hơn, EVAL_LIMIT=8 làm kết luận chỉ có giá trị sơ bộ: cần full eval trước khi quyết định triển khai.

---

## 6. Định tính — bắt buộc có cả ca THUA

Nhãn dưới đây đối chiếu từ `data/eval_target.jsonl`. Thứ tự nhãn là intent / urgency / product / sentiment. Log chỉ in dự đoán bị cắt và điểm fine-tune; không in dự đoán baseline từng mẫu. Vì vậy chưa thể xác nhận thắng/thua trực tiếp so với (b).

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 (i=3) | Bình giữ nhiệt; chưa thấy tiền; khi nào tiện | hoan_tien / thap / bình giữ nhiệt / tich_cuc | Chưa có dự đoán trong log | urgency=trung_binh; score=0.75 | Sai urgency so với nhãn; chưa biết có thua (b) không |
| 2 (i=0) | Chuột không dây; trả lại; gấp | doi_tra / cao / chuột không dây / tich_cuc | Chưa có dự đoán trong log | score=1.00 | Đúng đủ bốn trường theo điểm log |
| 3 (i=1) | Ốp lưng điện thoại; hoàn tiền; sớm nhé | hoan_tien / trung_binh / ốp lưng điện thoại / tieu_cuc | Chưa có dự đoán trong log | score=1.00 | Đúng đủ bốn trường theo điểm log |
| 4 (i=5) | Nồi chiên không dầu; thiếu phụ kiện; khi nào tiện | san_pham_loi / thap / nồi chiên không dầu / trung_tinh | Chưa có dự đoán trong log | score=1.00 | Đúng đủ bốn trường theo điểm log |
| 5 (i=6) | Balo laptop; đổi size; hỏi cho biết thôi | doi_tra / thap / balo laptop / tieu_cuc | Chưa có dự đoán trong log | score=1.00 | Đúng đủ bốn trường theo điểm log |

**Yêu cầu hai ca fine-tune thua baseline chưa được đáp ứng bằng bằng chứng hiện có.** Danh sách “3 ca tệ nhất” có hai ca score=1.0, nên không đồng nghĩa ba ca thua. Mẫu lỗi đã xác nhận là urgency: “Khi nào tiện” mang nhãn thấp nhưng model trả trung bình. Chưa đủ ca lỗi để khẳng định đây là mẫu chung. Cần tải `qualitative.json` và dự đoán baseline hoặc tạo so sánh từng mẫu trên full eval để hoàn thiện mục này; không suy diễn thêm dự đoán bị cắt.

---

## 7. Kết luận & điều tôi học được

Tôi chưa nên deploy adapter này. Dù điểm triage tăng mạnh so với base model dùng prompt tối ưu, cổng hồi quy vẫn FAILED và toàn bộ đánh giá chỉ là smoke test trên tám mẫu mỗi nhóm. Model có thể học rất tốt dạng JSON và các nhãn của tác vụ hẹp nhưng đồng thời giảm khả năng trả lời câu hỏi phổ thông. Vì thế, train loss thấp và format đúng không đủ để chứng minh chất lượng triển khai. Thí nghiệm khớp ngân sách tham số cho thấy `attn_only` có train loss tốt hơn nhưng target thấp hơn `correct`, nhắc tôi cần chọn mô hình bằng dữ liệu đánh giá thay vì chỉ số huấn luyện. LR cũng là đòn bẩy đáng kể: giảm một bậc với cùng ngân sách bước cho kết quả target và format bằng không trong run này. QLoRA tiết kiệm hơn một nửa VRAM nhưng mất điểm target và tăng thời gian, nên quyết định dùng nó phải gắn với giới hạn phần cứng thực tế. Mask đã được kiểm tra đúng và corpus cố định giúp phép so sánh có cơ sở, nhưng chưa loại bỏ vấn đề regression hoặc giới hạn kích thước eval. Bước tiếp theo là đo full eval, bổ sung so sánh định tính và thử replay data, rồi mới cân nhắc adapter cho triển khai.

**Ba điều tôi học được** (cụ thể, không generic):

1. So với (a) cho cảm giác thắng lớn, nhưng mốc có ý nghĩa là (b): prompt tối ưu đã đạt target 0.6875 và format 1.0 trong smoke test.
2. Với tham số gần như bằng nhau, loss 0.5379 của `attn_only` không thắng target 0.9688 của `correct`; vị trí adapter cần được kiểm tra bằng eval.
3. Tiết kiệm 4.92 GB bằng QLoRA đi kèm target giảm 0.125 và latency tăng; tối ưu bộ nhớ cần đi cùng kiểm tra chất lượng và tốc độ.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử:** chạy lại NB2 và NB5 với full eval trước, lưu đủ artefact và dự đoán; sau đó thử 1–5% replay data với cùng ngân sách bước và so sánh regression, không sửa tập eval hay hạ ngưỡng cổng để đạt PASS.

Gatekeeper trong log: 24 checks pass, 1 warning, 2 failures (report còn mẫu và eval bị rút gọn). Báo cáo này điền phần có bằng chứng; không chứng nhận gatekeeper đã pass sau chỉnh sửa. Repo local còn thiếu artefact NB2–NB5 và adapter `correct`. Bản nộp cần bổ sung các file thực từ Colab, full eval và bằng chứng định tính.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap — chưa có bằng chứng
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`) — dùng corpus mặc định
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`) — chưa làm cặp đối chứng
- [ ] B4 quét rank có kiểm soát — chỉ match rank cho placement, chưa quét rank
- [ ] B5 HuggingFace Hub — chưa upload adapter, chưa có link

Hugging Face đã được dùng để tải tokenizer ở NB1 và base model ở NB2–NB5 qua `from_pretrained`. Thưởng B5 là bước riêng: công khai adapter `adapters/correct/` lên Hub và ghi link trong report. Việc tải model từ Hub không đồng nghĩa đã hoàn thành B5.
