# Lab 2 (NumPy và Pandas Fundamentals):

## Nội dung chính
- Giới thiệu NumPy: thư viện cho tính toán số, nhanh hơn và hiệu quả hơn list Python; mảng đồng nhất, hỗ trợ phép toán vector hóa.
- Tạo mảng từ list và chuyển kiểu dữ liệu.

## Indexing & Slicing
- Truy cập phần tử: array[index]
- Cắt: array[start:end:step]
- Mảng nhiều chiều: array[row, col]
- Boolean indexing: array[condition]
- Fancy indexing: array[[i1, i2, ...]]

## Ví dụ thực hành
- 1D/2D: lấy hàng, cột, vùng con
- Các phép toán phức tạp với chỉ số
- Làm trò Tic-Tac-Toe bằng NumPy

## Xử lý dữ liệu với NumPy
- Lọc bằng điều kiện, kết hợp điều kiện với &&, || (np.logical_and, np.logical_or)
- Tạo dữ liệu mẫu để luyện tập

## Chuẩn bị dữ liệu cho ML
- Tạo dữ liệu tổng hợp (phân phối thực tế)
- Chia train/test 70/30
- Tạo 10 fold cross-validation