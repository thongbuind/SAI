// Phần hiển thị kết quả dùng chung cho mọi phòng làm việc: định dạng số, khối kính, nhãn mốc, bảng metric.
// Phần riêng của từng lệnh nằm trong views/. Chữ từ mô hình và từ spec luôn gán bằng textContent.
import type { Metric } from "./labApi";

// Tên metric do lab đặt → nhãn tiếng Việt; metric lạ hiện nguyên tên.
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

function fractionDigits(value: number): number {
  return Math.abs(value) >= LARGE_VALUE ? FRACTION_DIGITS_LARGE : FRACTION_DIGITS_SMALL;
}

export function formatNumber(value: number): string {
  return value.toLocaleString("vi-VN", { maximumFractionDigits: fractionDigits(value) });
}

// Làm tròn như formatNumber hiển thị, để hai ô cùng hiện một số thì cũng bằng nhau khi so ô tốt nhất.
export function roundAsDisplayed(value: number): number {
  const scale = 10 ** fractionDigits(value);
  return Math.round(value * scale) / scale;
}

export const formatSeconds = (ms: number) => `${formatNumber(ms / MS_PER_SECOND)} giây`;

export function cell(tag: "th" | "td", text: string, className?: string): HTMLTableCellElement {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}

// Nhãn trung tính cho cột, dòng hay mục dùng cấu hình mặc định hiện tại làm mốc.
export function baselineBadge(): HTMLElement {
  const badge = document.createElement("span");
  badge.className = "lab-base";
  badge.textContent = "mốc";
  return badge;
}

// Khối kính chứa một bảng hoặc một lưới kết quả; cuộn ngang khi nội dung rộng hơn vùng kết quả.
export function panel(label: string, content: HTMLElement): HTMLElement {
  const element = document.createElement("section");
  element.className = "lab-panel";
  element.setAttribute("aria-label", label);
  element.append(content);
  return element;
}

// Bảng metric: mỗi dòng một metric, mỗi cột một nhóm (biến thể); cột mốc mang nhãn "mốc". Ô tốt nhất theo "better"
// được tô, so trên số đã làm tròn như hiển thị nên các ô hoà nhau đều được tô.
export function metricsTable(metrics: Metric[], baseline: string | null = null): HTMLTableElement {
  const groups = [...new Set(metrics.map(metric => metric.group))];
  const names = [...new Set(metrics.map(metric => metric.name))];
  const table = document.createElement("table");
  table.className = "lab-metrics";
  const headRow = table.createTHead().insertRow();
  headRow.append(cell("th", ""), ...groups.map(group => {
    const head = cell("th", group);
    if (group === baseline) head.append(baselineBadge());
    return head;
  }));
  const body = table.createTBody();
  for (const name of names) {
    const row = body.insertRow();
    const sample = metrics.find(metric => metric.name === name)!;
    const label = METRIC_LABELS[name] ?? name;
    row.append(cell("th", sample.unit ? `${label} (${sample.unit})` : label));
    const values = groups.map(group => metrics.find(metric => metric.name === name && metric.group === group));
    const numbers = values.flatMap(metric => (metric ? [roundAsDisplayed(metric.value)] : []));
    const best = sample.better === "lower" ? Math.min(...numbers) : sample.better === "higher" ? Math.max(...numbers) : null;
    // Chỉ tô khi các nhóm khác nhau: một nhóm, hoặc mọi nhóm bằng nhau, thì không có gì để so.
    const comparable = groups.length > 1 && new Set(numbers).size > 1;
    for (const metric of values) {
      const isBest = metric !== undefined && comparable && roundAsDisplayed(metric.value) === best;
      row.append(cell("td", metric ? formatNumber(metric.value) : "—", isBest ? "is-best" : undefined));
    }
  }
  return table;
}
