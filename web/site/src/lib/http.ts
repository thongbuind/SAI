// Gọi API JSON cùng origin với site (web/backend: chat, bàn thử nghiệm). Mọi lỗi là RequestError có thông báo
// đọc được cho người dùng: ApiError (máy chủ trả mã lỗi) hoặc NetworkError (không tới được máy chủ).
import { url } from "./paths";

export class RequestError extends Error {}

export class ApiError extends RequestError {
  constructor(readonly status: number, message: string) {
    super(message);
  }
}

export class NetworkError extends RequestError {}

// statusText: câu viết sẵn cho mã lỗi mà FastAPI không trả detail dạng chuỗi (vd. 422 có detail là mảng).
async function errorText(response: Response, statusText: Record<number, string>): Promise<string> {
  if (response.headers.get("content-type")?.includes("application/json")) {
    const body = await response.json();
    if (typeof body.detail === "string") return body.detail;
  }
  return statusText[response.status] ?? `Máy chủ trả lỗi ${response.status}.`;
}

export async function requestJson<T>(path: string, init: RequestInit | undefined, statusText: Record<number, string>): Promise<T> {
  let response: Response;
  try {
    response = await fetch(url(path), init);
  } catch (error) {
    // fetch chỉ reject khi không có phản hồi HTTP (mất mạng, máy chủ tắt).
    throw new NetworkError("Không kết nối được máy chủ.", { cause: error });
  }
  if (!response.ok) throw new ApiError(response.status, await errorText(response, statusText));
  return response.json() as Promise<T>;
}

export const query = (params: Record<string, string>) => "?" + new URLSearchParams(params).toString();
export const jsonBody = (body: unknown): RequestInit => ({ method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
