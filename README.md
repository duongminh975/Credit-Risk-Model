# Credit Risk Scorecard — Mô hình xác suất vỡ nợ (Probability of Default)

Dự án xây dựng mô hình chấm điểm tín dụng khách hàng cá nhân (Retail Credit Scorecard) theo quy trình hoàn chỉnh, bao gồm phân nhóm dữ liệu bằng Weight of Evidence (WOE), hồi quy Logistic, chuyển đổi điểm tín dụng và phân tích danh mục cho vay bằng SQL.

**Công nghệ sử dụng:** Python (`pandas`, `NumPy`, `scikit-learn`) · Đánh giá thống kê (`ROC/AUC`, `Gini`, `KS`)

---

## Mô tả dự án

Các ngân hàng và tổ chức tài chính phi ngân hàng (NBFCs) sử dụng mô hình chấm điểm tín dụng để đưa ra quyết định cấp tín dụng. Đây là mô hình chuyển đổi các đặc điểm của người vay thành một điểm số duy nhất, từ đó xác định xác suất vỡ nợ tương ứng.

Dự án xây dựng mô hình theo các phương pháp tiêu chuẩn trong chấm điểm tín dụng khách hàng cá nhân (Retail Credit Scorecard) theo các bước sau:

1. **Coarse Classing:** Phân nhóm các giá trị của từng biến thành các khoảng hoặc nhóm phù hợp.
2. **Weight of Evidence (WOE):** Chuyển đổi các nhóm dữ liệu thành giá trị log-odds, sử dụng phương pháp hiệu chỉnh để xử lý các trường hợp có tần suất bằng 0.
3. **Information Value (IV):** Lựa chọn các biến dựa trên khả năng dự báo của chúng.
4. **Logistic Regression:** Xây dựng mô hình hồi quy Logistic có khả năng diễn giải trên các biến đã được chuyển đổi WOE.
5. **Points-based Scaling:** Chuyển đổi log-odds thành điểm số nguyên bằng phương pháp PDO.
6. **Validation:** Đánh giá mô hình thông qua AUC, Gini, KS Statistic và khả năng phân tầng rủi ro theo từng nhóm điểm (Decile Rank-ordering).

---

## Kết quả

### Hiệu suất mô hình (Model Performance)

| **Chỉ tiêu** | **Train (70%)** | **Test (30%)** |
|---|---:|---:|
| **AUC** | 0.8238 | 0.8000 |
| **Gini** | 0.6475 | 0.6000 |
| **KS Statistic** | 0.5231 | 0.5190 |

*Chênh lệch AUC giữa tập Train và Test là **0.024**, cho thấy mô hình có khả năng khái quát hóa tương đối tốt và không có dấu hiệu overfitting đáng kể.*

### Lựa chọn biến (Feature Selection)

20 biến được đánh giá, trong đó 16 biến được giữ lại với ngưỡng `IV ≥ 0.02`.

Các biến có khả năng dự báo mạnh nhất:

| **Biến** | **Information Value (IV)** | **Khả năng dự báo** |
|---|---:|---|
| Tình trạng tài khoản thanh toán (`status`) | 0.6300 | Rất mạnh |
| Lịch sử tín dụng (`credit_history`) | 0.2520 | Trung bình |
| Tiền tiết kiệm (`savings`) | 0.2213 | Trung bình |
| Thời hạn vay (`duration`) | 0.1838 | Trung bình |
| Tài sản sở hữu (`property`) | 0.1386 | Trung bình |
| Thời gian làm việc (`employment_duration`) | 0.1385 | Trung bình |

### Phân tầng rủi ro (Rank-Ordering trên tập Test)

Đây là bước kiểm định quan trọng nhằm đảm bảo tỷ lệ vỡ nợ tăng dần khi điểm tín dụng giảm xuống:

| **Nhóm điểm** | **Số lượng khoản vay** | **Tỷ lệ vỡ nợ thực tế** | **PD dự báo** |
|---|---:|---:|---:|
| **A (Tốt nhất)** | 60 | 6.7% | 5.0% |
| **B** | 57 | 14.0% | 12.1% |
| **C** | 59 | 17.0% | 23.1% |
| **D** | 64 | 43.8% | 40.8% |
| **E (Rủi ro cao nhất)** | 60 | 66.7% | 66.3% |

### Phân tích ngưỡng phê duyệt (Cut-Off Analysis)

| **Chính sách** | **Ngưỡng điểm** | **Tỷ lệ phê duyệt** | **Tỷ lệ nợ xấu** | **Số khoản vỡ nợ tránh được** |
|---|---:|---:|---:|---:|
| **Phê duyệt tất cả** | 424 | 100.0% | 30.0% | 0 |
| **Từ chối 10% nhóm thấp nhất** | 467 | 90.7% | 25.7% | 20 |
| **Từ chối 20% nhóm thấp nhất** | 486 | 80.0% | 20.8% | 40 |
| **Từ chối 30% nhóm thấp nhất** | 498 | 70.7% | 17.0% | 54 |
| **Từ chối 50% nhóm thấp nhất** | 521 | 50.0% | 10.7% | 74 |

*Từ chối 30% hồ sơ có điểm tín dụng thấp nhất giúp giảm tỷ lệ nợ xấu của danh mục từ 30% xuống còn 17%.*
