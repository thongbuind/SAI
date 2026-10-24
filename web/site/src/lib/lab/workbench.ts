// Trang bàn thử nghiệm theo phòng, chỉ có ở bản server: /lab/workbench/<department>/.
// command là lệnh của foundation_lab mà trang gọi (khoá trong "commands" của describe); description là dòng giới thiệu
// dưới tiêu đề trang.
export const WORKBENCH_PAGES = [
  { department: "inference", title: "Suy luận", command: "infer", description: "So sánh các cách giải mã trên cùng một checkpoint và cùng bộ câu hỏi." },
] as const;

export type WorkbenchPage = (typeof WORKBENCH_PAGES)[number];
