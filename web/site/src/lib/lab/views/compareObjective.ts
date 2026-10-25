// Phòng Objective (lệnh compare-objective): mỗi dòng một experiment, so objective ứng viên với objective mốc của stage
// trong experiment đó (sai khác loss và gradient, thời gian và tốc độ, bộ nhớ đỉnh) và kết luận khớp hay khác mốc.
// Objective và chunk của mốc lấy từ bản ghi lab ghi. Khung bảng dùng chung với phòng Kiến trúc nằm trong compareTable.ts.
import { ACTIVE_STATES } from "../labApi";
import { cell, formatNumber, formatSeconds, panel } from "../resultView";
import { comparisonTable, formatDiff, itemCell, itemNames, span, type ComparisonRecord } from "./compareTable";
import type { Spec, TaskView } from "./view";

interface ObjectiveRecord extends ComparisonRecord {
  experiment: string;
  stage: string;
  reference: string;
  candidate: string;
  reference_chunk_size: number;
  candidate_chunk_size: number;
  dtype: string;
  shape: { N: number; d_model: number; V: number };
  loss_rel_diff: number;
  max_abs_diff_grad_hidden: number;
  max_abs_diff_grad_weight: number;
}

// Dòng chính là sai khác tương đối của tổng loss, dòng phụ là sai khác lớn nhất trong gradient theo hidden và head
// weight; từng cái hiện khi rê chuột.
function diffCell(record: ObjectiveRecord): HTMLTableCellElement {
  const grads = [record.max_abs_diff_grad_hidden, record.max_abs_diff_grad_weight];
  const element = cell("td", `loss ${formatDiff(record.loss_rel_diff)}`);
  element.append(span("sub", `gradient ${formatDiff(Math.max(...grads))}`));
  element.title = `gradient hidden / weight: ${grads.map(formatDiff).join(" / ")}`;
  return element;
}

const OBJECTIVE_COLUMNS = {
  itemHead: "Experiment",
  diffHead: "Sai khác",
  itemName: (record: ObjectiveRecord) => record.experiment,
  itemCell(record: ObjectiveRecord) {
    const { d_model, V } = record.shape;
    // Objective và chunk ở đây là của mốc (tiêu đề khối đã ghi); bỏ chữ "mốc" để dòng phụ vừa cột ở màn laptop.
    const detail = `d${d_model} · V${formatNumber(V)} · ${record.reference}, chunk ${formatNumber(record.reference_chunk_size)}`;
    return itemCell(record.experiment, detail);
  },
  diffCell,
  mismatchText: "Khác mốc",
};

// Chunk ứng viên trống thì lab dùng loss_chunk_size của mốc trong từng experiment.
function chunkText(chunkSize: unknown): string {
  return typeof chunkSize === "number" ? `chunk ${formatNumber(chunkSize)}` : "chunk như mốc";
}

function tokensText(tokens: unknown): string {
  return typeof tokens === "number" ? `N = ${formatNumber(tokens)} token` : "N = max_microbatch_tokens của stage";
}

function summaryLine(config: Spec): HTMLElement {
  const line = document.createElement("p");
  line.className = "summary-line";
  const candidate = document.createElement("span");
  candidate.append("Ứng viên ", span("mono", String(config.candidate ?? "?")), ` · ${chunkText(config.candidate_chunk_size)}`);
  line.append(
    candidate,
    span("", `stage ${config.stage ?? "pretrain"}`),
    span("", `${config.dtype ?? "float32"} · ${config.device ?? "cpu"}`),
    span("", tokensText(config.tokens)),
    span("", `trung vị ${config.repeats ?? "?"} lần forward + backward`),
  );
  return line;
}

function experimentCount(experiments: unknown): string {
  return `${itemNames(experiments).length} experiment`;
}

function runLabel(config: Spec | null): string {
  return config ? `${config.candidate} · ${chunkText(config.candidate_chunk_size)} · ${experimentCount(config.experiments)}` : "";
}

export const objectiveView: TaskView = {
  workload: spec => `${experimentCount(spec.experiments)} · 2 objective mỗi experiment`,

  confirmText: spec =>
    `Lệnh sẽ so objective ${spec.candidate} (${chunkText(spec.candidate_chunk_size)}) với objective của stage ${spec.stage} ` +
    `trong ${experimentCount(spec.experiments)} (${itemNames(spec.experiments).join(", ")}), trên cỡ head của từng model, ` +
    `${tokensText(spec.tokens)}, ${spec.dtype} trên ${spec.device}.`,

  summary: runLabel,

  label: runLabel,

  timing: result => formatSeconds(result.timings.total_ms),

  progressUnit: "experiment",

  render(detail, slot) {
    const config = detail.config ?? {};
    const block = document.createElement("div");
    block.className = "block";
    const head = document.createElement("div");
    head.className = "block-head";
    const title = document.createElement("h2");
    title.className = "t-h4";
    title.textContent = "Theo experiment";
    head.append(title, span("t-caption", `mốc: params.objective và loss_chunk_size của stage ${config.stage ?? "pretrain"}`));

    const pendingText = ACTIVE_STATES.includes(detail.state) ? "Đang chờ" : "Không có kết quả";
    const { table, fill } = comparisonTable(OBJECTIVE_COLUMNS, itemNames(config.experiments), pendingText);

    const caption = document.createElement("p");
    caption.className = "t-caption";
    caption.textContent =
      "Objective trả về tổng loss trên N token nên sai khác loss tính tương đối theo mốc; rê chuột lên ô sai khác để " +
      "xem gradient theo hidden và theo head weight. Khớp theo dung sai mặc định của torch.testing.assert_close cho " +
      "kiểu số đã chọn. Objective khác về bản chất sẽ ra \"Khác mốc\": đó là kết quả mong đợi, xem độ lớn và chi phí. " +
      "Chunk nhỏ hơn giảm bộ nhớ của logits; bộ nhớ đỉnh chỉ đo được trên CUDA nên chỉ có cột này khi chạy trên CUDA. " +
      "Hợp đồng của objective nằm trong foundation_lab/objectives/registry.py.";
    block.append(head, summaryLine(config), panel("Theo experiment", table), caption);
    slot.replaceChildren(block);

    for (const record of detail.result?.records ?? []) fill(record as unknown as ObjectiveRecord);
    return record => fill(record as unknown as ObjectiveRecord);
  },
};
