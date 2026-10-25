// Phòng Suy luận (lệnh infer): bảng metric theo biến thể và lưới câu trả lời câu hỏi × biến thể. Ô câu hỏi mở sẵn
// benchmark của lab (foundation_lab/inference/configs/benchmark.yaml), có nút ghi đè benchmark khi đã sửa.
import { fetchBenchmark, saveBenchmark } from "../labApi";
import type { Control } from "../specForm";
import { baselineBadge, formatSeconds, metricsTable, panel } from "../resultView";
import { normalizeAIText } from "../../chatRender";
import { RequestError } from "../../http";
import type { Spec, TaskView } from "./view";

// Khoá câu hỏi trong spec của lệnh infer; ô của khoá này mở sẵn benchmark.
const PROMPTS_KEY = "prompts";
// Từ chừng này biến thể, cột câu trả lời giữ bề rộng tối thiểu và lưới cuộn ngang; ít hơn thì chia đều cho vừa khung.
const SCROLL_FROM_VARIANTS = 4;

const asList = (value: unknown) => (Array.isArray(value) ? value : []);
const basename = (path: unknown) => String(path ?? "").split("/").at(-1)!.replace(/\.[^.]+$/, "");

interface InferRecord {
  prompt_index: number;
  variant: string;
  text: string;
  finish_reason: string;
  completion_tokens: number;
  latency_ms: number;
}

// Thanh dưới ô câu hỏi: cho biết ô đang khớp benchmark của lab hay đã sửa; đã sửa thì có nút ghi đè benchmark.
function benchmarkBar(prompts: Control, saved: string[], showError: (message: string | null) => void): HTMLElement {
  let benchmark = saved;
  const bar = document.createElement("div");
  bar.className = "bench-benchmark";
  const status = document.createElement("span");
  status.className = "t-caption";
  const save = document.createElement("button");
  save.type = "button";
  save.className = "lab-link";
  save.textContent = "Lưu vào benchmark";
  bar.append(status, save);

  function render() {
    const isSaved = JSON.stringify(prompts.read()) === JSON.stringify(benchmark);
    status.textContent = isSaved ? `Benchmark của lab · ${benchmark.length} câu` : "Đã sửa, chưa lưu vào benchmark";
    bar.dataset.saved = String(isSaved);
    save.hidden = isSaved;
  }
  save.addEventListener("click", async () => {
    showError(null);
    save.disabled = true;
    try {
      benchmark = await saveBenchmark(prompts.read() as string[]);
    } catch (error) {
      if (!(error instanceof RequestError)) throw error;
      showError(error.message);
    } finally {
      save.disabled = false;
    }
    render();
  });
  prompts.element.addEventListener("input", render);
  render();
  return bar;
}

// Lưới so sánh: mỗi dòng một câu hỏi, mỗi cột một biến thể. Ô chờ hiện "Đang chờ" tới khi có bản ghi;
// fill() điền câu trả lời vào đúng ô (gọi được nhiều lần theo SSE).
function compareGrid(prompts: string[], variants: string[], baseline: string | null): { element: HTMLElement; fill(record: InferRecord): void } {
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
    if (variant === baseline) head.append(baselineBadge());
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

function generations(spec: Spec): number {
  return asList(spec.prompts).length * asList(spec.variants).length;
}

export const inferView: TaskView = {
  loadInitial: async () => ({ [PROMPTS_KEY]: await fetchBenchmark() }),

  decorateForm(form, fields, initial, showError) {
    const prompts = fields.querySelector(`[data-path="${PROMPTS_KEY}"]`)!;
    prompts.append(benchmarkBar(form.children![PROMPTS_KEY], initial[PROMPTS_KEY] as string[], showError));
  },

  workload: spec => `${generations(spec)} lượt sinh`,

  confirmText: spec =>
    `Lệnh sẽ nạp ${spec.model} với trọng số ${spec.checkpoint}, rồi sinh ${generations(spec)} câu trả lời ` +
    `(${asList(spec.prompts).length} câu hỏi × ${asList(spec.variants).length} biến thể).`,

  summary: config =>
    config ? `${basename(config.checkpoint)} · ${asList(config.prompts).length} câu × ${asList(config.variants).length} biến thể` : "",

  label: config => basename(config?.checkpoint),

  timing: result => `${formatSeconds(result.timings.total_ms)} · nạp ${formatSeconds(result.timings.load_ms)}`,

  progressUnit: "lượt",

  // Run đang chạy: dựng lưới theo câu hỏi × biến thể rồi điền dần từ SSE (luồng phát lại mọi sự kiện từ đầu).
  // Run đã xong: điền từ result.json.
  render(detail, slot) {
    const baseline = detail.result?.baseline ?? null;
    const blocks: HTMLElement[] = [];
    if (detail.result) blocks.push(panel("Metric", metricsTable(detail.result.metrics, baseline)));
    const prompts = asList(detail.config?.prompts).map(String);
    const variants = asList(detail.config?.variants).map(variant => String((variant as Spec).name));
    const grid = compareGrid(prompts, variants, baseline);
    if (prompts.length > 0) blocks.push(panel("Câu trả lời", grid.element));
    slot.replaceChildren(...blocks);
    for (const record of detail.result?.records ?? []) grid.fill(record as unknown as InferRecord);
    return record => grid.fill(record as unknown as InferRecord);
  },
};
