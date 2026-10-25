// Gọi API của phòng làm việc (web/backend/lab_routes.py, /api/lab). Kiểu dữ liệu ở đây là hợp đồng với lab:
// output "python -m foundation_lab describe" và file trong runs/<lệnh>/<run_id>/ (events.jsonl, result.json).
import { jsonBody, query, requestJson } from "../http";
import { url } from "../paths";

// Phải khớp DESCRIBE_FORMAT_VERSION trong foundation_lab/cli.py (web/backend/lab_cli.py cũng kiểm tra lúc khởi động).
export const DESCRIBE_FORMAT_VERSION = 1;

// Mô tả kiểu của foundation_lab.core.config.describe_type.
interface FieldMeta { default?: unknown; required?: boolean; nullable?: boolean }
export type ObjectSchema = FieldMeta & { type: "object"; name: string; fields: Record<string, FieldSchema> };
export type FieldSchema = FieldMeta & (
  | ObjectSchema
  | { type: "list"; items: FieldSchema }
  | { type: "map"; values: FieldSchema }
  | { type: "union"; options: FieldSchema[] }
  | { type: "choice"; values: (string | number | boolean)[] }
  | { type: "bool" | "int" | "float" | "str" | "any" }
);

export interface Registry { kind: string; names: string[] }
export interface Description {
  format_version: number;
  lab_version: string;
  departments: Record<string, Registry[]>;
  commands: Record<string, { department: string; spec: ObjectSchema }>;
  models: string[];
  // Tên model → họ mô hình (vd. SAI, SAI_modular).
  model_families: Record<string, string>;
  // Experiment chạy được → các stage đang bật, theo thứ tự pipeline.
  experiments: Record<string, string[]>;
  // Model → trọng số nạp được vào model đó (chỉ model đang có trọng số trên máy).
  checkpoints: Record<string, string[]>;
}

export type RunState = "queued" | "running" | "succeeded" | "failed" | "cancelled" | "incomplete";
export const ACTIVE_STATES: RunState[] = ["queued", "running"];

export interface LabEvent { seq: number; time: number; kind: "started" | "progress" | "record" | "finished" | "failed"; [key: string]: unknown }
export interface Metric { group: string; name: string; value: number; unit: string; better: "lower" | "higher" | "none" }
export interface RunResult {
  format_version: number;
  command: string;
  metrics: Metric[];
  records: Record<string, unknown>[];
  timings: Record<string, number>;
  // Group dùng cấu hình mặc định làm mốc; null khi mốc nằm trong từng bản ghi. result.json cũ không có khoá này.
  baseline?: string | null;
}
export interface RunSummary { command: string; run_id: string; state: RunState; config: Record<string, unknown> | null }
export interface RunDetail extends RunSummary { events: LabEvent[]; result: RunResult | null; stderr: string[] }

const STATUS_TEXT: Record<number, string> = {
  403: "Phòng làm việc chỉ mở được từ localhost.",
  422: "Yêu cầu không hợp lệ.",
};

const request = <T>(path: string, init?: RequestInit) => requestJson<T>(path, init, STATUS_TEXT);

export const fetchDescription = () => request<Description>("/api/lab/describe");

export const listRuns = (command: string) =>
  request<{ runs: RunSummary[] }>("/api/lab/jobs" + query({ command })).then(body => body.runs);

export const getRun = (command: string, runId: string) => request<RunDetail>(`/api/lab/jobs/${command}/${runId}`);

export const submitJob = (command: string, spec: Record<string, unknown>) =>
  request<{ command: string; run_id: string }>("/api/lab/jobs", jsonBody({ command, spec })).then(body => body.run_id);

export const cancelJob = (command: string, runId: string) =>
  request<{ state: RunState }>(`/api/lab/jobs/${command}/${runId}/cancel`, { method: "POST" });

// Bộ câu hỏi mặc định của phòng Suy luận, lưu trong lab (benchmark_path của describe).
export const fetchBenchmark = () => request<{ prompts: string[] }>("/api/lab/benchmark").then(body => body.prompts);

export const saveBenchmark = (prompts: string[]) =>
  request<{ prompts: string[] }>("/api/lab/benchmark", { ...jsonBody({ prompts }), method: "PUT" }).then(body => body.prompts);

// Theo dõi sự kiện của một run qua SSE. Server kết thúc luồng bằng sự kiện "end" kèm trạng thái cuối;
// đóng EventSource ngay lúc đó để trình duyệt không tự kết nối lại.
export function watchRun(
  command: string, runId: string,
  onEvent: (event: LabEvent) => void, onEnd: (state: RunState) => void,
): () => void {
  const source = new EventSource(url(`/api/lab/jobs/${command}/${runId}/events`));
  for (const kind of ["started", "progress", "record", "finished", "failed"]) {
    source.addEventListener(kind, message => onEvent(JSON.parse((message as MessageEvent).data)));
  }
  source.addEventListener("end", message => {
    source.close();
    onEnd(JSON.parse((message as MessageEvent).data).state);
  });
  return () => source.close();
}
