// Trang phòng làm việc theo phòng, ở cả hai bản (bản static chỉ xem): /lab/workbench/<department>/, theo thứ tự phòng
// trong docstring của foundation_lab. department là mã phòng trong docstring đó (vd. arch), cũng là phòng mà describe
// ghi cho lệnh của phòng; description là dòng giới thiệu dưới tiêu đề trang.
// - Phòng đã chạy được từ web có command: lệnh của foundation_lab mà trang gọi (khoá trong "commands" của describe,
//   phần riêng của lệnh nằm trong views/), và empty: câu đầu của trạng thái chưa có lần chạy.
// - Phòng chưa có lệnh là trang chờ (components/lab/PlannedBench.astro), có planned: lệnh dự kiến của phòng, null khi
//   phòng không cần lệnh riêng.
export const WORKBENCH_PAGES = [
  {
    department: "core", title: "Ban điều hành", planned: "validate --json",
    description: "Kiểm tra một experiment trước khi chạy: config đã resolve, các stage và những khoá khác cấu hình mặc định.",
  },
  {
    department: "arch", title: "Kiến trúc", command: "compare-kernel",
    description: "So một attention kernel với kernel mặc định của model: sai khác output và gradient, tốc độ, bộ nhớ đỉnh.",
    empty: "Chọn kernel ứng viên và các model ở control center rồi bấm Chạy.",
  },
  {
    department: "data", title: "Dữ liệu", planned: "inspect-data",
    description: "Xem dữ liệu mà từng stage sẽ đọc: số mẫu, độ dài, token chịu loss và hiệu suất padding theo đúng cách chia batch của stage.",
  },
  {
    department: "tokenizer", title: "Tokenizer", planned: "compare-tokenizer",
    description: "So tokenizer ứng viên với tokenizer chính thức trên cùng bộ văn bản: số token, độ nén, byte fallback.",
  },
  {
    department: "objectives", title: "Objective", command: "compare-objective",
    description: "So một hàm loss với loss mà stage đang dùng: giá trị, gradient, tốc độ và bộ nhớ đỉnh, trên đúng cỡ head của model.",
    empty: "Chọn objective ứng viên, các experiment và stage làm mốc ở control center rồi bấm Chạy.",
  },
  {
    department: "engine", title: "Vận hành", planned: "bench-step",
    description: "Đo tốc độ và bộ nhớ của một bước train với các tuỳ chọn chạy khác nhau, so với cấu hình mặc định của experiment.",
  },
  {
    // Kho run: web đọc thẳng runs/, không gọi lệnh lab.
    department: "state", title: "Lưu trữ", planned: null,
    description: "Mọi run trên máy: config, lineage, trạng thái từng stage và trọng số.",
  },
  {
    department: "recipes", title: "Phương pháp huấn luyện", planned: "run",
    description: "Chạy một experiment, theo dõi loss của từng stage và so với run dùng cấu hình mặc định.",
  },
  {
    department: "eval", title: "Đánh giá", planned: "eval-loss",
    description: "Tính val/test loss của các checkpoint trên cùng dữ liệu của một stage, so với trọng số phát hành.",
  },
  {
    department: "inference", title: "Suy luận", command: "infer",
    description: "So sánh các cách giải mã trên cùng một checkpoint và cùng bộ câu hỏi.",
    empty: "Chọn checkpoint, câu hỏi và vài biến thể ở control center rồi bấm Chạy.",
  },
] as const;

export type WorkbenchPage = (typeof WORKBENCH_PAGES)[number];
