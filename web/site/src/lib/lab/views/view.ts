// Phần riêng của một lệnh lab trong phòng làm việc. Khung chung (components/lab/TaskBench.astro) lo form, lịch sử
// run, trạng thái, tiến trình, huỷ, dòng lệnh chạy lại; view lo cách tóm tắt một run và cách vẽ kết quả của nó.
import type { RunDetail, RunResult } from "../labApi";
import type { Control } from "../specForm";

export type Spec = Record<string, unknown>;

export interface TaskView {
  // Giá trị ban đầu của form mà lab giữ (vd. benchmark của infer), theo khoá cấp cao nhất của spec.
  loadInitial?(): Promise<Spec>;
  // Gắn thêm vào form đã dựng (vd. thanh "Lưu vào benchmark" dưới ô câu hỏi). showError hiện lỗi ở chân control center.
  decorateForm?(form: Control, fields: HTMLElement, initial: Spec, showError: (message: string | null) => void): void;
  // Khối lượng của spec đang nhập, hiện ở chân control center (vd. "40 lượt sinh").
  workload(spec: Spec): string;
  // Câu mô tả việc lệnh sẽ làm, trong hộp xác nhận trước khi chạy.
  confirmText(spec: Spec): string;
  // Tóm tắt một run từ config của nó (menu lịch sử; thanh đầu khi chưa có kết quả).
  summary(config: Spec | null): string;
  // Phần sau ngày giờ trên nút chọn lần chạy.
  label(config: Spec | null): string;
  // Thời gian khi run đã có result.json.
  timing(result: RunResult): string;
  // Đơn vị tiến trình trên thanh đầu khi đang chạy (vd. "lượt" trong "3/40 lượt").
  progressUnit: string;
  // Vẽ kết quả của run vào slot; trả về hàm điền một bản ghi từ SSE vào đúng chỗ khi run đang chạy.
  render(detail: RunDetail, slot: HTMLElement): (record: Spec) => void;
}
