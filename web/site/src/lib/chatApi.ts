// Gọi API của bản server (web/backend, cùng origin với site). Lỗi khi gọi API: RequestError trong lib/http.ts.
import { jsonBody, query, requestJson } from "./http";

export { ApiError, NetworkError, RequestError } from "./http";

export type ModelState = "pending" | "loading" | "ready" | "error";
export interface ModelStatus { name: string; state: ModelState }
export interface ModelList { models: ModelStatus[]; default: string }
export interface ChatReply { reply: string; model: string; latency_ms: number }

// Định dạng file lịch sử do backend lưu; giữ "ai" và "system" như chat cũ để mở được lịch sử cũ.
export interface StoredMessage { role: "user" | "ai" | "system"; text: string; model?: string }
export interface Session { id: string; title: string; messages: StoredMessage[]; created_at: number; user_id: string; shared: boolean }
export interface SessionSummary { id: string; title: string; created_at: number; shared: boolean }
export interface CommunitySummary { id: string; title: string; created_at: number; user_id: string; model: string | null }

// Phải khớp MAX_MESSAGE_CHARS trong web/backend/chat_routes.py.
export const MAX_MESSAGE_CHARS = 2000;

const STATUS_TEXT: Record<number, string> = {
  422: `Tin nhắn trống hoặc dài quá ${MAX_MESSAGE_CHARS.toLocaleString("vi-VN")} ký tự.`,
  502: "Máy chủ không phản hồi.",
};

// Mã ẩn danh của trình duyệt; cùng khoá với chat cũ để người dùng cũ còn thấy lịch sử của mình.
const USER_ID_KEY = "sai_user_id";

// getRandomValues chạy được cả khi trang mở qua http (randomUUID chỉ có trong secure context).
function randomId(prefix: string): string {
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return prefix + Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
}

export function browserUserId(): string {
  const created = randomId("u");
  try {
    const saved = localStorage.getItem(USER_ID_KEY);
    if (saved) return saved;
    localStorage.setItem(USER_ID_KEY, created);
  } catch {
    // Trình duyệt chặn lưu trữ: mã chỉ sống trong lần mở trang này, lịch sử vẫn lưu nhưng không mở lại được.
  }
  return created;
}

export function newSessionId(): string {
  return randomId("s");
}

const request = <T>(path: string, init?: RequestInit) => requestJson<T>(path, init, STATUS_TEXT);

export const fetchModels = () => request<ModelList>("/api/models");

export const sendChat = (message: string, userId: string, modelName: string, sessionId: string) =>
  request<ChatReply>("/api/chat", jsonBody({ message, user_id: userId, model_name: modelName, session_id: sessionId }));

export const listHistory = (userId: string) =>
  request<{ sessions: SessionSummary[] }>("/api/history" + query({ user_id: userId })).then(body => body.sessions);

export const listCommunity = () =>
  request<{ sessions: CommunitySummary[] }>("/api/history/community").then(body => body.sessions);

export const getSession = (id: string, userId: string) =>
  request<Session>(`/api/history/${id}` + query({ user_id: userId }));

export const saveSession = (session: Session) => request<{ ok: true }>("/api/history/save", jsonBody(session));

export const toggleShare = (id: string, userId: string) =>
  request<{ ok: true; shared: boolean }>(`/api/history/${id}/share` + query({ user_id: userId }), { method: "PATCH" }).then(body => body.shared);

export const deleteSession = (id: string, userId: string) =>
  request<{ ok: true }>(`/api/history/${id}` + query({ user_id: userId }), { method: "DELETE" });
