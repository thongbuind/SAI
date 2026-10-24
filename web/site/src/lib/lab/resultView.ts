// Hiển thị kết quả run của lab. Chữ từ mô hình và từ spec luôn gán bằng textContent.
import type { Metric } from "./labApi";
import { normalizeAIText } from "../chatRender";

// Tên metric do lab đặt (foundation_lab/inference/bench.py) → nhãn tiếng Việt; metric lạ hiện nguyên tên.
const METRIC_LABELS: Record<string, string> = {
  latency_ms_mean: "Thời gian trung bình",
  tokens_per_second: "Tốc độ sinh",
  completion_tokens_mean: "Độ dài trung bình",
  truncated_rate: "Tỷ lệ bị cắt",
  distinct_2: "Distinct-2",
};
const FRACTION_DIGITS_LARGE = 0;
const FRACTION_DIGITS_SMALL = 2;
const LARGE_VALUE = 100;
const MS_PER_SECOND = 1000;
// Từ chừng này biến thể, cột câu trả lời giữ bề rộng tối thiểu và lưới cuộn ngang; ít hơn thì chia đều cho vừa khung.
const SCROLL_FROM_VARIANTS = 4;

export function formatNumber(value: number): string {
  const digits = Math.abs(value) >= LARGE_VALUE ? FRACTION_DIGITS_LARGE : FRACTION_DIGITS_SMALL;
  return value.toLocaleString("vi-VN", { maximumFractionDigits: digits });
}

export const formatSeconds = (ms: number) => `${formatNumber(ms / MS_PER_SECOND)} giây`;

function cell(tag: "th" | "td", text: string, className?: string): HTMLTableCellElement {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

// Bảng metric: mỗi dòng một metric, mỗi cột một nhóm (biến thể); ô tốt nhất theo "better" được tô.
export function metricsTable(metrics: Metric[]): HTMLTableElement {
  const groups = [...new Set(metrics.map(metric => metric.group))];
  const names = [...new Set(metrics.map(metric => metric.name))];
  const table = document.createElement("table");
  table.className = "lab-metrics";
  const headRow = table.createTHead().insertRow();
  headRow.append(cell("th", ""), ...groups.map(group => cell("th", group)));
  const body = table.createTBody();
  for (const name of names) {
    const row = body.insertRow();
    const sample = metrics.find(metric => metric.name === name)!;
    const label = METRIC_LABELS[name] ?? name;
    row.append(cell("th", sample.unit ? `${label} (${sample.unit})` : label));
    const values = groups.map(group => metrics.find(metric => metric.name === name && metric.group === group));
    const numbers = values.flatMap(metric => (metric ? [metric.value] : []));
    const best = sample.better === "lower" ? Math.min(...numbers) : sample.better === "higher" ? Math.max(...numbers) : null;
    // Chỉ tô khi các nhóm khác nhau: một nhóm, hoặc mọi nhóm bằng nhau, thì không có gì để so.
    const comparable = groups.length > 1 && new Set(numbers).size > 1;
    for (const metric of values) {
      const isBest = metric !== undefined && comparable && metric.value === best;
      row.append(cell("td", metric ? formatNumber(metric.value) : "—", isBest ? "is-best" : undefined));
    }
  }
  return table;
}

export interface InferRecord {
  prompt_index: number;
  variant: string;
  text: string;
  finish_reason: string;
  completion_tokens: number;
  latency_ms: number;
}

// Lưới so sánh: mỗi dòng một câu hỏi, mỗi cột một biến thể. Ô chờ hiện "Đang chờ" tới khi có bản ghi;
// fill() điền câu trả lời vào đúng ô (gọi được nhiều lần theo SSE).
export function compareGrid(prompts: string[], variants: string[]): { element: HTMLElement; fill(record: InferRecord): void } {
  const grid = document.createElement("div");
  grid.className = "lab-compare";
  grid.style.setProperty("--variants", String(variants.length));
  grid.dataset.scroll = String(variants.length >= SCROLL_FROM_VARIANTS);
  const corner = document.createElement("div");
  corner.className = "lab-compare-head";
  corner.textContent = "Câu hỏi";
  grid.append(corner, ...variants.map(variant => {
    const head = document.createElement("div");
    head.className = "lab-compare-head";
    head.textContent = variant;
    return head;
  }));

  const cells = new Map<string, HTMLElement>();
  const key = (promptIndex: number, variant: string) => `${promptIndex}\u0000${variant}`;
  prompts.forEach((prompt, promptIndex) => {
    const promptCell = document.createElement("div");
    promptCell.className = "lab-compare-prompt";
    promptCell.textContent = prompt;
    grid.append(promptCell);
    for (const variant of variants) {
      const answer = document.createElement("div");
      answer.className = "lab-answer is-pending";
      answer.dataset.variant = variant;
      answer.textContent = "Đang chờ";
      cells.set(key(promptIndex, variant), answer);
      grid.append(answer);
    }
  });

  function fill(record: InferRecord) {
    const answer = cells.get(key(record.prompt_index, record.variant));
    if (!answer) return;
    answer.classList.remove("is-pending");
    const text = document.createElement("p");
    text.className = "lab-answer-text";
    text.textContent = normalizeAIText(record.text);
    const meta = document.createElement("p");
    meta.className = "lab-answer-meta";
    meta.textContent = `${record.completion_tokens} token · ${formatSeconds(record.latency_ms)} · ${record.finish_reason}`;
    answer.replaceChildren(text, meta);
  }
  return { element: grid, fill };
}
