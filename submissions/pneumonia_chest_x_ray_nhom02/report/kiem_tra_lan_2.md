# Bài kiểm tra lần 2 — <02> — <pneumonia_chest_x_ray>

- Fork: <https://github.com/lamtran91919/hmyt-pneumonia> · Nhánh: gd5-proposed-model
- Thành viên (MSSV hoặc alias) và vai trò: 
20233963: Xây dựng đánh giá mô hình, đánh giá kế quả
20224394: Xử lý dữ liệu, hoàn thiện báo cáo

## A. Literature Review và research gaps

### A1. Chiến lược tìm kiếm
| Nguồn (PubMed, Scopus, IEEE Xplore, Google Scholar…) | Từ khóa / truy vấn | Khoảng năm | Số kết quả | Số bài giữ lại |
|PubMed|("Pneumonia" OR "pneumonia detection") AND ("chest X-ray" OR "chest radiograph") AND ("deep learning" OR "machine learning" OR CNN OR transformer) |2018-2026|---|---| 
|IEEE Xplore|("pneumonia" AND ("chest X-ray" OR "chest radiograph")) AND ("deep learning" OR CNN OR transformer OR "domain adaptation")|2021-2026|---|---|
|Google Scholar|"pneumonia detection" "chest X-ray" deep learning external validation|2021-2026|---|---|

Tiêu chí chọn / loại bài:
- Một bài báo được giữ lại khi đáp ứng phần lớn các tiêu chí sau:
* Liên quan đến sử dụng ảnh X-quang ngực để phát hiện hoặc phân loại viêm phổi và có kết quả cho phân loại Normal/Pneumonia 
* Mô tả mô hình, phương pháp huấn luyện hoặc quy trình đánh giá đủ để xác định cách nghiên cứu được thực hiện.
* Có kết quả đánh giá: Báo cáo ít nhất một metric như Accuracy, AUROC, Precision, Recall, F1-score, Sensitivity hoặc Specificity.
* Có nguồn dữ liệu xác định: Nêu được dataset hoặc nguồn ảnh, chẳng hạn Guangzhou/Kermany, NIH ChestX-ray14, CheXpert hoặc các bộ dữ liệu CXR công khai khác.
* Ưu tiên bài có Transfer Learning, đánh giá trên nhiều nguồn dữ liệu, kiểm tra domain shift, shortcut learning hoặc explainability.
* Ưu tiên bài báo khoa học đã xuất bản, có DOI hoặc đường dẫn đến nhà xuất bản/kho lưu trữ học thuật.
* Loại các bài không nghiên cứu phát hiện viêm phổi trên ảnh, hoặc chỉ nghiên cứu một vấn đề không trực tiếp phục vụ phân loại Normal/Pneumonia
* Loại bài không đủ thông tin để thẩm định: Không xác định được nguồn dữ liệu, cách đánh giá hoặc kết quả chính.
* Loại bài không phù hợp với phạm vi ma trận chỉ giới thiệu kiến trúc nền tảng hoặc phương pháp giải thích, không trực tiếp thực hiện phân loại pneumonia trên CXR, được giữ để tham khảo phần Background/Methods nhưng không tính vào ma trận nghiên cứu ứng dụng A2.

### A2. Ma trận y văn (≥ 10 bài, ưu tiên 3–5 năm gần nhất)
| # | Bài (tác giả, năm) | DOI / link | Dataset (n, nguồn) | Phương pháp | Kết quả chính (metric, giá trị) | Kiểm định ngoài? (có / không) | Hạn chế chính |
|1|Kermany et al. (2018), Cell|https://doi.org/10.1016/j.cell.2018.02.010|5,863 ảnh CXR trẻ em từ Guangzhou Women and Children’s Medical Center|Transfer learning với Inception-V3 pretrained; phân loại ảnh CXR Normal/Pneumonia.|Accuracy 92.8%; sensitivity 93.2%; specificity 90.1% trên test set|Không|Không chứng minh khả năng tổng quát hóa sang người lớn hoặc bệnh viện khác.|
|2|Zech et al. (2018), PLOS Medicine|https://doi.org/10.1371/journal.pmed.1002683|158,323 radiographs: NIH 112,120 (30,805 bệnh nhân), Mount Sinai 42,396 (12,904), Indiana University 3,807 (3,683)|CNN/DenseNet-based pneumonia screening; so sánh train/test nội bộ và ngoại kiểm giữa ba hệ thống bệnh viện|Mô hình kết hợp NIH–MSH: internal AUC 0.931; external IU AUC 0.815. Chỉ dùng hospital system để xếp hạng đã đạt AUC 0.861 trên NIH–MSH.|Có|Nhãn pneumonia khác nhau theo nguồn; tỷ lệ bệnh khác biệt lớn giữa bệnh viện; bài chỉ ra confounding theo bệnh viện.|
|3|Chouhan et al. (2020), Applied Sciences|https://doi.org/10.3390/app10020559|5,232 ảnh CXR từ Guangzhou Women and Children’s Medical Center|Transfer learning với AlexNet, DenseNet121, InceptionV3, GoogLeNet và ResNet18; kết hợp dự đoán bằng majority voting.|Ensemble: Accuracy 96.39%; sensitivity/recall 99.62%; precision 93.28%; AUC 99.34%.|Không|Một nguồn dữ liệu bệnh nhi; đánh giá trên split nội bộ; không chứng minh external generalization.|
|4|Rahman et al. (2020), Applied Sciences|https://doi.org/10.3390/app10093233|Kaggle Chest X-ray Pneumonia dataset, khoảng 5.2 nghìn ảnh; chủ yếu CXR trẻ em|So sánh AlexNet, ResNet18, DenseNet201 và SqueezeNet bằng transfer learning; augmentation; ba bài toán Normal/Pneumonia, bacterial/viral và ba lớp.|DenseNet201: Accuracy 98% (Normal/Pneumonia), 95% (bacterial/viral), 93.3% (3 lớp); bài báo báo cáo Precision 97% và Recall 99% cho bài toán nhị phân.|Không|Một bộ dữ liệu công khai, chủ yếu bệnh nhi; không có kiểm định bệnh viện độc lập; các metric không nên gộp/so sánh trực tiếp.|
|5|Salehi et al. (2021), British Journal of Radiology|https://doi.org/10.1259/bjr.20201263|5,863 ảnh CXR trẻ em từ Guangzhou Women and Children’s Medical Center|So sánh VGG19, Xception, ResNet50, DenseNet121 pretrained; augmentation/SMOTE; dùng Grad-CAM để trực quan hóa vùng ảnh.|DenseNet121 tốt nhất trong bốn mô hình: AUC 0.86; Accuracy 86.8%; Precision 87.0%; Sensitivity 92.8%; F1 89.8%.|Không|Tập dữ liệu đơn nguồn và bệnh nhi; kết quả nội bộ; không chứng minh tổng quát hóa ngoài nguồn.|
|6|Wang et al. (2021), Nature Biomedical Engineering|https://doi.org/10.1038/s41551-021-00704-1|Dataset phát triển 145,202 CXR đa trung tâm; đánh giá thêm hàng nghìn ảnh ở bốn cohort và nhiều quốc gia; có bộ độc lập 440 ảnh để so sánh với bác sĩ|Chuẩn hóa CXR, phát hiện bệnh lý lồng ngực, định vị/phân đoạn phổi và tổn thương, phân loại pneumonia/COVID-19 và đánh giá mức độ nặng.|AUC 0.94–0.98 cho phân biệt viral pneumonia/other pneumonia/no disease; AUC 0.87–0.97 cho COVID-19 pneumonia so với các pneumonia khác.|Có|Nhiều module và cần annotation chuyên gia; mục tiêu rộng hơn binary pneumonia/normal|
|7|Trivedi & Gupta (2022), Multimedia Tools and Applications|https://doi.org/10.1007/s11042-021-11807-x|Dataset-1: 5,856 ảnh CXR trẻ em từ Guangzhou Women and Children’s Medical Center. Dataset-2 độc lập: 5,500 ảnh train + 325 validation + 325 test.|Kiến trúc nhẹ dựa trên MobileNet/depthwise separable convolution; oversampling và augmentation; huấn luyện ở Dataset-1 rồi đánh giá Dataset-2.|94.23% trên Dataset-1; 82.46% khi chuyển sang Dataset-2 mà không train lại; 90.15% sau khi train trên Dataset-2. External test specificity 0.71, precision 0.83, recall 0.89, F1 0.86.|Có|Hiệu năng giảm rõ khi chuyển nguồn; hiệu năng phục hồi khi train trên nguồn đích|
|8|Mabrouk et al. (2022), Applied Sciences|https://doi.org/10.3390/app12136448|5,856 ảnh CXR trẻ em từ Guangzhou Women and Children’s Medical Center|Feature-level ensemble kết hợp DenseNet169, MobileNetV2 và ViT pretrained/fine-tuned; Adam và early stopping.|Ensemble: Accuracy 93.91%; Precision 93.96%; Recall 92.99%; F1 93.43%.|Không|Một nguồn bệnh nhi; test split nội bộ; không có ngoại kiểm; ensemble cần cân nhắc tính tái lập.|
|9|Ukwuoma et al. (2023; online 2022), Journal of Advanced Research|https://doi.org/10.1016/j.jare.2022.08.021|Hai nguồn công khai: Mendeley Chest X-ray và Chest X-ray-15K (tập đa lớp tổng hợp từ 11 nguồn)|Hai cấu hình ensemble CNN nối với Transformer Encoder, multi-head attention và MLP; saliency maps.|Binary: Accuracy và F1 99.21%; 3-class: Accuracy 98.19%, F1 97.29%.|Không|Kiến trúc ensemble-transformer phức tạp; saliency map không chứng minh nguyên nhân bệnh lý; chưa xác nhận external validation|
|10|Singh et al. (2024), Scientific Reports|https://doi.org/10.1038/s41598-024-52703-2|Public Kaggle/Mendeley pediatric CXR dataset; test set 586 ảnh theo confusion matrix được báo cáo|Vision Transformer pretrained; patch embedding và Transformer Encoder; phân loại Normal/Pneumonia.|Accuracy 97.61%; sensitivity 95%; specificity 98%; AUC khoảng 0.96; MCC khoảng 0.9396.|Không|Một benchmark công khai; không có external hospital cohort; kết quả không loại trừ shortcut learning hoặc domain shift.|
|11|Ali et al. (2024), IEEE Access|https://doi.org/10.1109/ACCESS.2024.3372588|5,856 ảnh CXR trẻ em từ Guangzhou Women and Children’s Medical Center|So sánh CNN, InceptionResNetV2, Xception, VGG16, ResNet50 và EfficientNetV2L; Adam và augmentation.|EfficientNetV2L: Accuracy 94.02%; Precision 94.40%; Recall 97.24%; F1 95.80%.|Không|Một bộ dữ liệu đơn; so sánh tập trung vào accuracy/benchmark; không có ngoại kiểm độc lập được báo cáo|
|12|Yanar et al. (2025), Applied Sciences – PELM|https://doi.org/10.3390/app15126487|50,000 ảnh PA sau lọc/cân bằng, lấy từ NIH ChestX-ray14, CheXpert, PadChest và Kaggle CXR Pneumonia; 40k train, 5k validation, 5k test; bài mô tả giữ ảnh PA đầu tiên mỗi bệnh nhân|Feature-level concatenation của InceptionV3, VGG16, ResNet50 và ViT; ImageNet transfer learning, batch normalization và dropout.|Accuracy 96%; Precision 99%; Recall 91%; F1 95%; Specificity 91%; AUC 0.91.|Không|Lọc bỏ ảnh chất lượng kém và cân bằng 1:1 làm giảm tính đại diện thực tế; hạn chế đại diện tuổi/bệnh nền; ensemble khó giải thích|

### A3. Research gaps
| Mã | Khoảng trống (1–2 câu) | Bằng chứng (số # trong A2) | Dự án của nhóm giải quyết bằng cách nào |
| G1 | Hầu hết nghiên cứu trên bộ Guangzhou báo cáo hiệu năng cao trên cùng một nguồn, không kiểm định ngoài; khi có kiểm định ngoài thì hiệu năng giảm.| Không ngoại kiểm: #1, #8, #10, #11, #15, #16, #17. Giảm khi ngoại kiểm: #5, #9 | Huấn luyện ở một nguồn, test ở nguồn kia (Guangzhou sang NIH và ngược lại) |   
| G2 | 	Tín hiệu không liên quan phổi (nguồn dữ liệu, viền, kích thước, nhãn máy) có thể làm tăng hiệu năng nhưng hiếm khi được đo lường trên bộ Guangzhou. | #5, #17 | Probe thống kê đơn giản (đoán nhãn và nguồn từ thống kê ảnh), ghép cặp theo tỉ lệ khung hình và viền đen, đo lại sau ghép; kiểm tra che vùng trung tâm |   
| G3 | Rò rỉ giữa train và test do chia ngẫu nhiên theo ảnh, không theo bệnh nhân, ít được kiểm soát. | #10; các bài #8, #11, #16 không nêu cách chia | Chia theo bệnh nhân, riêng cho từng nguồn; kiểm tra ảnh gần trùng giữa các tập |
| G4 | Explainability thường được trình bày dưới dạng Grad-CAM/saliency nhưng chưa chứng minh mô hình phụ thuộc đúng vùng bệnh lý hoặc có khả năng giải thích đáng tin cậy. | #5, #9, #10, #13 | Dùng Grad-CAM để kiểm tra định tính và kết hợp masking/ablation vùng ảnh để kiểm tra độ nhạy của dự đoán. Nêu rõ Grad-CAM là công cụ hỗ trợ phân tích, không phải ground truth. |
| G5 | Nhãn từ các bộ dữ liệu khác nhau có thể được tạo bằng quy trình khác nhau (NLP từ report, phân loại công khai, chuyên gia xác nhận), dẫn đến label heterogeneity và khác biệt định nghĩa outcome. | #2, #6, #12 | Mô tả nguồn và cách gán nhãn của Guangzhou/NIH; phân tích kết quả theo dataset để không che lấp chênh lệch. |
| G6 | Nhiều bài nhấn mạnh Accuracy/F1 nhưng chưa luôn trình bày đầy đủ trade-off giữa sensitivity và specificity, độ bất định thống kê và tính ổn định theo nguồn. | #3, #4, #8, #10, #11, #12, #13 | Báo cáo AUROC, Accuracy, Precision, Recall, F1, Sensitivity và Specificity; sử dụng patient-level bootstrap 95% CI cho AUROC nếu có thể. |
| G7 | Các benchmark công khai thường có phân bố bệnh nhân hẹp hoặc đã lọc ảnh chất lượng thấp, nên áp dụng cho môi trường lâm sàng còn hạn chế. | #1, #3, #4, #5, #8, #10, #11, #12 | Thảo luận rõ giới hạn của hai nguồn dữ liệu hiện có và không tuyên bố mô hình tổng quát cho mọi độ tuổi. Xem đây là giới hạn cần nghiên cứu tiếp nếu chưa có cohort đa trung tâm có nhãn chuẩn. |
### A4. Thẩm định 1 bài quan trọng nhất trong A2 bằng PROBAST+AI
Bài số 5: Zech et al. (2018), PLOS Medicine · Lý do chọn:
| Miền | Nguy cơ sai lệch (thấp / cao / không rõ) | Căn cứ (trích dẫn hoặc số trang trong bài) |
|---|---|---|
| Người tham gia và nguồn dữ liệu | Không rõ | 158.323 ảnh từ 3 hệ thống bệnh viện, thiết kế cắt ngang hồi cứu. Quần thể khác nhau: tuổi trung bình 46,9 / 63,2 / 49,6, nữ 43,5% / 44,8% / 57,3%; tỉ lệ viêm phổi 1,2% (NIH), 34,2% (MSH), 1,0% (IU). |
| Biến dự báo | Thấp | Dự báo là ảnh X-quang, giống nhau giữa các nguồn. Chính bài cho thấy ảnh chứa thông tin nhận ra bệnh viện (99,95% NIH, 99,98% MSH) và khoa trong bệnh viện. |
| Kết cục | Không rõ | Kết cục là "dấu hiệu X-quang phù hợp viêm phổi" (radiographic findings consistent with pneumonia), không phải chẩn đoán lâm sàng. |
| Phân tích | Thấp | Dùng split-sample validation với nhiều tập huấn luyện, so sánh AUC nội và ngoại, thí nghiệm lấy mẫu phân tầng để tách riêng ảnh hưởng của tỉ lệ bệnh; chưa rõ việc tinh chỉnh tham số và xử lý dữ liệu thiếu; không thể đánh giá hết các yếu tố khác gây thiên lệch theo bệnh viện. |
Tính áp dụng cho bài toán của nhóm: Cao về ý tưởng, trung bình về bối cảnh. Kết quả biện minh cho việc nhóm cân bằng tỉ lệ nhãn theo nguồn, báo cáo theo từng nguồn và kiểm định chéo nguồn. Khác biệt là bài dùng ba hệ thống bệnh viện người lớn, còn nhóm dùng Guangzhou (nhi, một trung tâm) và NIH (một trung tâm, nhãn khai phá bằng NLP).

## B. Kết quả và timeline

### B1. Kết quả hiện có
| Mô hình | Tập đánh giá (cách chia, n) | Metric chính (giá trị, 95% CI nếu có) | Metric phụ | Nguồn số liệu (file / notebook / W&B) |
|---|---|---|---|---|
| Baseline 1 (Mốc 2) | Test 1.087 ảnh (Guangzhou 808, NIH 279); chia theo bệnh nhân nhưng gộp hai nguồn | NIH AUROC 0,802 [0,747–0,857]; Guangzhou 0,994 [0,991–0,997]; gộp 0,971 [0,961–0,979] | NIH: sens 0,688, spec 0,767; Guangzhou: sens 0,969, spec 0,964 | runs/guangzhou-nih_resizefirst_matchABS_s2/results.json và notebook ô EVALUATION, bootstrap, dashboard |
| Mô hình đề xuất (Mốc 3) | Chia theo bệnh nhân, riêng từng nguồn (~71/14/14): train 2.865, val 562, test 555 (NIH 286, Guangzhou 269) | NIH AUROC 0,748 [0,689–0,802]; Guangzhou AUROC 0,988 [0,979–0,996]; gộp 0,903 [0,877–0,928] | NIH: sens 0,562, spec 0,795; Guangzhou: sens 0,945, spec 0,958 (ngưỡng riêng: NIH 0,577, Guangzhou 0,644) | runs/guangzhou-nih_resizefirst_matchAB_s2/results.json và notebook ô EVALUATION, bootstrap, dashboard |

### B2. Ablation (Mốc 3)
| Cấu hình (bỏ / đổi thành phần nào) | Metric chính | Chênh lệch so với mô hình đầy đủ | Nguồn |
|Chưa thực hiện|Chưa có|Chưa có|Chưa có|

### B3. Đối chiếu với y văn
So sánh kết quả B1 với 2–3 bài trong A2. Nêu rõ những điểm làm phép so sánh **không công bằng**, nếu có: khác dataset, khác cách chia, khác metric.

#9 Trivedi & Gupta, 2022: Bài đạt Accuracy 94,23% in-source; 82,46% trên dataset ngoài không train lại, so với nhóm được AUROC Guangzhou 0,985 (in-source) so với NIH 0,675 (ngoại). So sánh không cân bằng do khác metric (accuracy so với AUROC); dataset ngoài khác (nhóm dùng NIH người lớn, định nghĩa nhãn khác); bài không dùng chia theo bệnh nhân

#5 Zech et al., 2018: Bài đạt AUC nội 0,931; ngoại (Indiana) 0,815; so với nhóm đạt nội 0,985; ngoại 0,675 (Guangzhou sang NIH). Mức giảm của nhóm lớn hơn (−0,31 so với −0,12), có thể do chuyển từ nhi sang người lớn, nhãn khác loại (lâm sàng so với khai phá NLP), mô hình chỉ 1.417 ảnh Guangzhou sau ghép; bài của Zech dùng hàng chục nghìn ảnh người lớn

#8 Ali et al., 2024: Bài đạt EfficientNetV2L Accuracy 94,02% trên 5.856 ảnh Guangzhou, so với Guangzhou Accuracy ≈ 95,2% (tính lại từ ma trận nhầm lẫn: 256/269) ở ngưỡng riêng	của nhóm. Tập test của nhóm là tập con đã ghép cặp (269 ảnh, 34% dữ liệu), chia theo bệnh nhân; bài của Ali chia khác và chỉ báo cáo Accuracy trên toàn bộ dữ liệu

### B4. Timeline đến khi hoàn thành
| Việc | Phụ trách (MSSV / vai trò) | Mốc (M3 tuần 10 · M4 tuần 12 · Final tuần 15) | Ngày dự kiến | Trạng thái (Chưa / Đang / Xong + bằng chứng) |
|---|---|---|---|---|

| Pipeline dữ liệu: đọc ảnh, ghép cặp, chia theo bệnh nhân từng nguồn  | 20224394 (data) + 20233963 | M3 | 9/11/2026 | Xong: notebook ô BUILD ONE DATAFRAME và PATIENT-LEVEL SPLIT |
| Huấn luyện mô hình đề xuất, 4 run (hai nguồn × 2 biến thể ghép cặp, Guangzhou-only, NIH-only) | 20233963 | M3 | 9/11/2026  | Xong (1 seed): runs/*/results.json |
| Kiểm định chéo nguồn (ngoại) | 20233963 | M3 | 9/11/2026  | Xong (1 seed): 2 run Guangzhou-only, NIH-only |
| Ablation tiền xử lý (CLAHE, cân bằng histogram, tắt ghép cặp) | 20233963 | M4 | 22/11/2026 | Chưa |
| Chạy lặp ≥ 3 seed cho cấu hình cuối | 20233963 | M4 | 22/11/2026  | Chưa (ô SEED NOISE cần ≥ 2 run cùng cấu hình) |
| Baseline hồi quy logistic + calibration (đường cong, in-the-large, slope) + Brier | 20233963 | M4 | 22/11/2026  | Chưa |
| Literature review: kiểm chứng số liệu 18 bài, bổ sung số trang A4 | 20224394 | M3 | 9/11/2026  | Đang: 18 bài đã lập, cần sửa các   điểm |
| Datasheet, checksum, giấy phép dữ liệu | 20224394 | M4 | 22/11/2026  | Chưa |
| Điền checklist TRIPOD+AI | 20224394 | M4 | 22/11/2026  | Chưa |
| Báo cáo cuối và thuyết trình | 20224394 + 20233963 | Final (tuần 15) | Tuần cuối  | Chưa |
### B5. Rủi ro / đang vướng / câu hỏi cho GV
- Tập test nhỏ (NIH 286, Guangzhou 269): khoảng tin cậy AUROC rộng khoảng ±0,055
- Guangzhou vẫn đạt AUROC 0,988 sau ghép cặp. Khi che vùng trung tâm ngực, Guangzhou còn 0,797 (NIH chỉ còn 0,526), nên còn khả năng mô hình đọc thông tin ngoài phổi ở Guangzhou; chưa phân biệt được tín hiệu thật và tín hiệu rác.
- Nhãn NIH khai phá bằng NLP và lớp âm là "No Finding", nên bài toán có thể là "bất thường so với bình thường" chứ chưa hẳn đặc hiệu viêm phổi; chưa có thử nghiệm với ảnh bất thường khác (hard negative).
- Mô hình huấn luyện trên Guangzhou chuyển sang NIH chỉ đạt 0,675, nên chưa thể tuyên bố khái quát hóa.
Các run matchABS và matchAB dùng hai biến thể ghép cặp; notebook hiện tại chỉ có mức AB. Cần ghi rõ "S" là gì và đưa mã tương ứng vào repo.
- Ô bootstrap in khoảng tin cậy sens / spec ở ngưỡng chung, không khớp với ngưỡng riêng theo nguồn đang dùng.
- Câu hỏi cho GV: 
(1) Kiểm định chéo nguồn (Guangzhou và NIH) có được tính là kiểm định ngoài không
(2) Hướng dẫn báo cáo phù hợp là TRIPOD+AI hay STARD-AI; 
(3) Với bài toán ảnh, "mô hình mốc hồi quy logistic" nên dùng đặc trưng nào; 

## C. Tự đánh giá dự án theo Phụ lục FAIR, TRIPOD+AI, PROBAST+AI

### C1. FAIR — dữ liệu
| Mục | Trả lời | Bằng chứng |
|---|---|---|
| Nguồn, phiên bản, ngày tải; checksum file dùng để huấn luyện | Guangzhou: bản trên Kaggle paultimothymooney/chest-xray-pneumonia, bản sao của bộ Kermany et al. (2018); nhóm tìm thấy 5.855 ảnh (bản gốc 5.856, thiếu 1 ảnh). NIH: bản đóng gói lại chehablab/NIHChestXR trên Hugging Face (ảnh 224 px) của NIH ChestX-ray14; subset 2.000 ảnh (1.000 viêm phổi, 1.000 No Finding) chọn đều ngẫu nhiên bằng prepare_nih_subset.py (seed 42). Ngày tải bộ Guangzhou: 20/9/2026 | |
| Giấy phép / DUA, căn cứ được phép dùng | Hai dataset public | Guangzhou: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data; NIH: https://huggingface.co/datasets/chehablab/NIHChestXR/viewer|
| Datasheet: quần thể, tiêu chí chọn/loại, định nghĩa và đơn vị biến, giới hạn | Guangzhou: trẻ em < 5 tuổi, một trung tâm; nhãn Normal/Pneumonia; sau ghép cặp giữ 1.982 ảnh (991 mỗi lớp). NIH: 112.120 ảnh của 30.805 bệnh nhân, tuổi trung bình 46,9 (theo Zech et al., 2018); nhãn khai phá bằng NLP từ báo cáo; nhóm dùng 2,000 ảnh (1,000 có nhãn Pneumonia và 1,000 No Finding). Biến: ảnh xám 8 bit (0–255), nhãn nhị phân. Giới hạn: hai quần thể và định nghĩa nhãn khác nhau; ảnh NIH đã bị thu nhỏ về 224 px; nhãn NIH nhiễu. | prepare_nih_subset.py; notebook ô BUILD ONE DATAFRAME |
| Không có định danh bệnh nhân | Hai bộ dữ liệu công khai đã ẩn danh bởi nơi cung cấp; pipeline chỉ dùng mã bệnh nhân ẩn danh để gom ảnh cùng bệnh nhân khi chia tập; chữ nhãn in trên ảnh (L / R, "PORTABLE") vẫn còn trong ảnh | Notebook ô PATIENT-LEVEL SPLIT |

### C2. Hướng dẫn báo cáo
- Câu hỏi nghiên cứu của nhóm: Mô hình học sâu phát hiện viêm phổi trên X-quang ngực (EfficientNet-B0 tinh chỉnh) có giữ được hiệu năng khi chuyển từ nguồn Guangzhou (trẻ em) sang NIH (chủ yếu người lớn) không, và tín hiệu không liên quan phổi (viền, kích thước, tỉ lệ khung hình, nguồn) đóng góp bao nhiêu vào hiệu năng đó?
- Hướng dẫn báo cáo phù hợp (TRIPOD+AI / STARD-AI / DECIDE-AI) và lý do (Phụ lục §3.2): TRIPOD+AI, vì đây là nghiên cứu phát triển và kiểm định mô hình chẩn đoán dựa trên học máy, có kiểm định nội bộ và chéo nguồn. STARD-AI phù hợp hơn nếu mục tiêu chính là độ chính xác chẩn đoán của một xét nghiệm AI trong quần thể lâm sàng xác định; DECIDE-AI dành cho đánh giá lâm sàng giai đoạn sớm, không áp dụng.
- Số mục checklist đã đáp ứng / tổng số mục; các mục còn thiếu: 19/27 mục, thiếu: protocol và kế hoạch phân tích, calibration, fairness giữa các nhóm bệnh nhân, đánh giá prospective hoặc tác động của mô hình, khả năng tái lập, và patient identifiers
- Kết cục, nhãn, thời điểm dự đoán T0, các biến có sẵn tại T0: Kết cục: có viêm phổi hay không (nhị phân). Nhãn Guangzhou do bác sĩ gán; nhãn NIH khai phá bằng NLP từ báo cáo đọc phim. T0 là vào thời điểm chụp ảnh X-quang ngực; mô hình chẩn đoán trên một ảnh, không có khoảng dự báo theo thời gian. Biến có sẵn tại T0: chỉ ảnh (điểm ảnh); không dùng biến lâm sàng. Mã bệnh nhân chỉ dùng để chia tập, không đưa vào mô hình.

### C3. PROBAST+AI — tự thẩm định dự án của nhóm
| Miền | Nguy cơ sai lệch (thấp / cao / không rõ) | Căn cứ trong repo (file, dòng) | Việc sẽ làm để giảm nguy cơ |
|---|---|---|---|
| Người tham gia và nguồn dữ liệu | Cao | Hai bộ dữ liệu công khai, mỗi bộ một trung tâm (Guangzhou: trẻ em; NIH: người lớn); NIH chỉ lấy subset 2.000 ảnh; Guangzhou giữ 1.982 / 5.855 ảnh sau ghép cặp. Ô BUILD ONE DATAFRAME, prepare_nih_subset.py | Thêm nguồn thứ ba độc lập nếu có thể; hoàn thiện datasheet; báo cáo theo từng nguồn |
| Biến dự báo | Cao | Ảnh chứa tín hiệu ngoài phổi: sau ghép cặp, thống kê ảnh đơn giản vẫn đoán nhãn Guangzhou với AUROC 0,866; che vùng trung tâm vẫn còn 0,797 (Guangzhou). Ô probe, ô centre-masking | Ghép cặp (đã làm một phần); thử cân bằng histogram; cắt theo vùng phổi bằng phân đoạn; Grad-CAM; đo lại |
| Kết cục (chú ý incorporation bias) | Cao | Nhãn NIH khai phá từ báo cáo đọc phim (nhãn phụ thuộc vào chính việc đọc ảnh); lớp âm chỉ là "No Finding". Ô BUILD ONE DATAFRAME | Thử nghiệm ảnh có bệnh lý khác; cân nhắc nguồn nhãn do chuyên gia đọc; nêu giới hạn trong báo cáo |
| Phân tích (chia theo bệnh nhân, fit chỉ trên train, cỡ mẫu, dữ liệu thiếu) | Không rõ | Đã chia theo bệnh nhân riêng từng nguồn; CLAHE và resize xử lý từng ảnh, không fit trên toàn bộ dữ liệu; chọn ngưỡng và epoch trên validation; CI bootstrap theo bệnh nhân; kiểm tra ảnh gần trùng. Còn hạn chế: các khoảng chia của ghép cặp tính trên toàn bộ ảnh trước khi chia tập; từng so sánh cấu hình bằng kết quả test; test nhỏ (±0,055) | Tính lại ghép cặp chỉ từ tập train hoặc nêu rõ giới hạn; chọn cấu hình theo validation; chạy ≥ 3 seed; thêm calibration và Brier; loại cặp ảnh gần trùng khỏi train |
Tính áp dụng (quần thể, bối cảnh dùng mô hình): Thấp ở thời điểm này. Mô hình đạt AUROC 0,985–0,988 trên Guangzhou nhưng chỉ 0,675 (Guangzhou sang NIH) và 0,660 (NIH sang Guangzhou) khi sang nguồn khác; chưa kiểm chứng lâm sàng.

### C4. Bảng kiểm nhanh (Phụ lục §7)
| Mục | Đạt / Chưa | Bằng chứng |
|---|---|---|
| Datasheet | Chưa | Có bản nháp, chưa có tài liệu riêng trong repo |
| Phiên bản + checksum dữ liệu | Chưa | Mới có số lượng ảnh (5.855 / 5.856 Guangzhou); chưa có phiên bản và SHA-256 |
| Không định danh; căn cứ được phép dùng dữ liệu | Đạt | Dữ liệu công khai đã ẩn danh; dữ liệu public |
| Chọn đúng hướng dẫn báo cáo + checklist đã điền | Chưa | Chưa điền checklist |
| Định nghĩa kết cục, nhãn, T0 | Đạt | C2 |
| Chia theo bệnh nhân; tiền xử lý fit chỉ trên train | Đạt | Chia theo bệnh nhân (ô PATIENT-LEVEL SPLIT), có tiền xử lý từng ảnh.|
| Mô hình mốc hồi quy logistic (hoặc mốc phù hợp, nêu lý do) | Chưa | Mới có probe HistGradientBoosting trên thống kê ảnh (không cùng tập test) và lần chạy đầu làm mốc tạm (B1) |
| AUC + calibration (đường cong, in-the-large, slope) + Brier, kèm 95% CI | Chưa | Thiếu Calibration và Brier|
| Cỡ mẫu có đủ không; giới hạn về tính áp dụng | Chưa | Không có tính toán cỡ mẫu |

## D. Khai báo dùng AI
| Công cụ | Dùng cho phần nào | Nhóm đã kiểm chứng lại bằng cách nào |
| Claude | Hỗ trợ thiết kế và viết notebook (tiền xử lý, chia tập, đánh giá, dashboard), gợi ý cách diễn giải kết quả, soạn bản nháp file này | Chạy lại notebook và so số liệu; đối chiếu số liệu y văn với toàn văn từng bài; kiểm tra lại link, DOI và số trang |
| ChatGPT | Hỗ trợ giải thích các metric và tổng hợp nội dung các bài báo | Đọc lại và so sánh phần tóm tắt với các bài báo được cung cấp cũng như đối chiếu với các bài báo khác |