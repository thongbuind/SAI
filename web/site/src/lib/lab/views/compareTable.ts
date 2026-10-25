// Bảng so ứng viên với mốc theo từng mục, dùng chung cho phòng Kiến trúc (mục là model) và phòng Objective (mục là
// experiment): mỗi dòng một mục gồm sai khác (phần riêng của từng lệnh), thời gian và tốc độ, bộ nhớ đỉnh, kết luận.
// Bảng giữ ít cột để vừa cột giữa của trang ở màn laptop, không phải cuộn ngang mới thấy kết luận.
import { cell, formatNumber, roundAsDisplayed } from "../resultView";

// Số cột số đo sau cột tên mục (kể cả cột bộ nhớ khi đang ẩn); dòng chưa có bản ghi gộp chúng thành một ô.
const MEASURE_COLUMNS = 4;
// Bảng mang lớp này thì hiện cột bộ nhớ đỉnh. Chỉ CUDA đo được, nên trên CPU/MPS bỏ hẳn cột thay vì một cột toàn "—".
const MEMORY_CLASS = "has-memory";

// Phần chung trong bản ghi của các lệnh so sánh (xem _run_comparisons trong foundation_lab/eval/compare.py).
export interface ComparisonRecord {
  matches_reference: boolean;
  reference_ms: number;
  candidate_ms: number;
  speedup: number;
  reference_peak_mib: number | null;
  candidate_peak_mib: number | null;
}

// Phần riêng của từng lệnh trong bảng.
export interface ComparisonColumns<R> {
  itemHead: string;
  diffHead: string;
  // Nhãn của mục trong bản ghi, để điền bản ghi vào đúng dòng.
  itemName(record: R): string;
  itemCell(record: R): HTMLTableCellElement;
  diffCell(record: R): HTMLTableCellElement;
  // Nhãn kết luận khi ứng viên không khớp mốc.
  mismatchText: string;
}

// Config của run: danh sách mục là danh sách tên (spec gửi đi) hoặc mapping tên → config (resolved_config.yaml).
export function itemNames(items: unknown): string[] {
  if (Array.isArray(items)) return items.map(String);
  return items !== null && typeof items === "object" ? Object.keys(items) : [];
}

// Sai khác rất nhỏ nên viết dạng mũ, dấu phẩy thập phân như mọi số khác: 2,4e-7.
export function formatDiff(value: number): string {
  return value === 0 ? "0" : value.toExponential(1).replace(".", ",");
}

export function span(className: string, text: string): HTMLSpanElement {
  const element = document.createElement("span");
  element.className = className;
  element.textContent = text;
  return element;
}

function headCell(text: string, alignLeft = false): HTMLTableCellElement {
  return cell("th", text, alignLeft ? "l" : undefined);
}

// Ô tên mục: tên và một dòng phụ (cỡ, mốc; hoặc trạng thái khi chưa có bản ghi).
export function itemCell(name: string, detail: string): HTMLTableCellElement {
  const head = cell("th", name);
  head.append(span("sub", detail));
  return head;
}

// Hai thanh cùng thang trong một ô: thời gian của mốc (trung tính) và của ứng viên (tím).
function ratioBars(reference: number, candidate: number): HTMLElement {
  const bars = document.createElement("span");
  bars.className = "lab-ratio";
  bars.setAttribute("aria-hidden", "true");
  const longest = Math.max(reference, candidate);
  for (const [className, value] of [["r-base", reference], ["r-cand", candidate]] as const) {
    const bar = document.createElement("span");
    bar.className = className;
    bar.style.width = `${(value / longest) * 100}%`;
    bars.append(bar);
  }
  return bars;
}

function speedDelta(speedup: number): HTMLSpanElement {
  const shown = roundAsDisplayed(speedup);
  const delta = span("lab-delta", `×${formatNumber(speedup)}`);
  if (shown !== 1) delta.dataset.trend = shown > 1 ? "better" : "worse";
  return delta;
}

// Thời gian mốc → ứng viên kèm tỉ lệ tốc độ, hai thanh cùng thang ngay bên dưới.
function timeCell(record: ComparisonRecord): HTMLTableCellElement {
  const element = cell("td", `${formatNumber(record.reference_ms)} → ${formatNumber(record.candidate_ms)}`);
  element.append(speedDelta(record.speedup), ratioBars(record.reference_ms, record.candidate_ms));
  return element;
}

function memoryCell(record: ComparisonRecord): HTMLTableCellElement {
  const text = record.reference_peak_mib === null || record.candidate_peak_mib === null
    ? "—"
    : `${formatNumber(record.reference_peak_mib)} → ${formatNumber(record.candidate_peak_mib)}`;
  return cell("td", text, "memory");
}

function verdictCell(matches: boolean, mismatchText: string): HTMLTableCellElement {
  const verdict = cell("td", "", "l");
  const badge = span("lab-state", matches ? "Khớp" : mismatchText);
  badge.dataset.state = matches ? "match" : "mismatch";
  verdict.append(badge);
  return verdict;
}

// Tiêu đề cột thời gian kèm chú giải màu của hai thanh.
function timeHead(): HTMLTableCellElement {
  const head = headCell("Thời gian (ms)");
  const legend = document.createElement("span");
  legend.className = "lab-legend";
  for (const [className, text] of [["k-base", "mốc"], ["k-1", "ứng viên"]] as const) {
    const item = document.createElement("span");
    const key = document.createElement("i");
    key.className = `key ${className}`;
    item.append(key, text);
    legend.append(item);
  }
  head.append(legend);
  return head;
}

// Bảng có sẵn một dòng cho mỗi mục trong names (pendingText khi chưa có bản ghi); fill thay cả dòng khi bản ghi về.
export function comparisonTable<R extends ComparisonRecord>(
  columns: ComparisonColumns<R>, names: string[], pendingText: string,
): { table: HTMLTableElement; fill: (record: R) => void } {
  const table = document.createElement("table");
  table.className = "lab-metrics rows";
  table.createTHead().insertRow().append(
    headCell(columns.itemHead), headCell(columns.diffHead), timeHead(), cell("th", "Bộ nhớ đỉnh (MiB)", "memory"),
    headCell("Kết luận", true),
  );
  const body = table.createTBody();
  const rows = new Map<string, HTMLTableRowElement>();
  for (const name of names) {
    const row = body.insertRow();
    row.append(itemCell(name, pendingText), cell("td", pendingText, "pending"));
    row.cells[1].colSpan = MEASURE_COLUMNS;
    rows.set(name, row);
  }

  function fill(record: R) {
    if (record.reference_peak_mib !== null) table.classList.add(MEMORY_CLASS);
    rows.get(columns.itemName(record))?.replaceChildren(
      columns.itemCell(record), columns.diffCell(record), timeCell(record), memoryCell(record),
      verdictCell(record.matches_reference, columns.mismatchText),
    );
  }
  return { table, fill };
}
