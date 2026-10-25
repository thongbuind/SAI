// Phòng Kiến trúc (lệnh compare-kernel): mỗi dòng một model, so kernel ứng viên với kernel mốc của model đó (sai khác
// output và gradient, thời gian và tốc độ, bộ nhớ đỉnh) và kết luận khớp hay lệch. Kernel mốc lấy từ bản ghi lab ghi.
// Khung bảng dùng chung với phòng Objective nằm trong compareTable.ts.
import { ACTIVE_STATES } from "../labApi";
import { cell, formatSeconds, panel } from "../resultView";
import { comparisonTable, formatDiff, itemCell, itemNames, span, type ComparisonRecord } from "./compareTable";
import type { Spec, TaskView } from "./view";

interface KernelRecord extends ComparisonRecord {
  model: string;
  reference: string;
  candidate: string;
  dtype: string;
  shape: { B: number; H: number; Hkv: number; T: number; D: number };
  max_abs_diff_out: number;
  max_abs_diff_grad_q: number;
  max_abs_diff_grad_k: number;
  max_abs_diff_grad_v: number;
}

// Dòng chính là sai khác output, dòng phụ là sai khác lớn nhất trong gradient q, k, v; từng cái hiện khi rê chuột.
function diffCell(record: KernelRecord): HTMLTableCellElement {
  const grads = [record.max_abs_diff_grad_q, record.max_abs_diff_grad_k, record.max_abs_diff_grad_v];
  const element = cell("td", `output ${formatDiff(record.max_abs_diff_out)}`);
  element.append(span("sub", `gradient ${formatDiff(Math.max(...grads))}`));
  element.title = `gradient q / k / v: ${grads.map(formatDiff).join(" / ")}`;
  return element;
}

const KERNEL_COLUMNS = {
  itemHead: "Mô hình",
  diffHead: "Sai khác lớn nhất",
  itemName: (record: KernelRecord) => record.model,
  itemCell(record: KernelRecord) {
    const { H, Hkv, D } = record.shape;
    return itemCell(record.model, `H${H} · Hkv${Hkv} · D${D} · mốc ${record.reference}`);
  },
  diffCell,
  mismatchText: "Lệch",
};

function summaryLine(config: Spec): HTMLElement {
  const line = document.createElement("p");
  line.className = "summary-line";
  const candidate = document.createElement("span");
  candidate.append("Ứng viên ", span("mono", String(config.candidate ?? "?")));
  line.append(
    candidate,
    span("", `${config.dtype ?? "float32"} · ${config.device ?? "cpu"}`),
    span("", `B=${config.batch_size ?? "?"} · T=${config.seq_len ?? "?"}${config.causal === false ? "" : " · causal"}`),
    span("", `trung vị ${config.repeats ?? "?"} lần forward + backward`),
  );
  return line;
}

function modelCount(models: unknown): string {
  return `${itemNames(models).length} model`;
}

export const kernelView: TaskView = {
  workload: spec => `${modelCount(spec.models)} · 2 kernel mỗi model`,

  confirmText: spec =>
    `Lệnh sẽ so kernel ${spec.candidate} với kernel mặc định của ${modelCount(spec.models)} (${itemNames(spec.models).join(", ")}), ` +
    `trên shape attention của từng model, ${spec.dtype} trên ${spec.device}.`,

  summary: config => (config ? `${config.candidate} · ${modelCount(config.models)}` : ""),

  label: config => (config ? `${config.candidate} · ${modelCount(config.models)}` : ""),

  timing: result => formatSeconds(result.timings.total_ms),

  progressUnit: "model",

  render(detail, slot) {
    const config = detail.config ?? {};
    const block = document.createElement("div");
    block.className = "block";
    const head = document.createElement("div");
    head.className = "block-head";
    const title = document.createElement("h2");
    title.className = "t-h4";
    title.textContent = "Theo cỡ mô hình";
    head.append(title, span("t-caption", "mốc: attention_kernel trong config của từng model"));

    const pendingText = ACTIVE_STATES.includes(detail.state) ? "Đang chờ" : "Không có kết quả";
    const { table, fill } = comparisonTable(KERNEL_COLUMNS, itemNames(config.models), pendingText);

    const caption = document.createElement("p");
    caption.className = "t-caption";
    caption.textContent =
      "Kernel không có tham số: khớp nghĩa là đổi kernel mà vẫn dùng nguyên trọng số đã train. Khớp theo dung sai " +
      "mặc định của torch.testing.assert_close cho kiểu số đã chọn. Rê chuột lên ô sai khác để xem gradient theo từng " +
      "q, k, v. Bộ nhớ đỉnh chỉ đo được trên CUDA nên chỉ có cột này khi chạy trên CUDA.";
    block.append(head, summaryLine(config), panel("Theo cỡ mô hình", table), caption);
    slot.replaceChildren(block);

    for (const record of detail.result?.records ?? []) fill(record as unknown as KernelRecord);
    return record => fill(record as unknown as KernelRecord);
  },
};
