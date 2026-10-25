// Dựng phần tử DOM cho trang chat (tin nhắn, mục lịch sử) và định dạng chữ. Chữ người dùng và mô hình
// luôn gán bằng textContent; chỉ SVG icon (hằng số trong icons.ts) đi qua innerHTML.
import { ICONS, type IconName } from "./icons";
import models from "../data/models.json";

export function iconSvg(name: IconName): string {
  return `<svg class="icon" width="16" height="16" viewBox="0 0 24 24" aria-hidden="true">${ICONS[name]}</svg>`;
}

export function iconButton(icon: IconName, label: string): HTMLButtonElement {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "icon-btn icon-btn-sm";
  button.setAttribute("aria-label", label);
  button.innerHTML = iconSvg(icon);
  return button;
}

// "SAI_35M" → "SAI 35M". Lịch sử cũ lưu tên file checkpoint (vd. "sft1_35M.pt") thì hiện nguyên văn.
export function modelLabel(name: string): string {
  const model = models.find(m => m.name === name);
  return model ? `SAI ${model.label}` : name;
}

// Mô hình sinh chữ thường (tokenizer huấn luyện trên dữ liệu chữ thường). Viết hoa chữ cái đầu câu và đầu dòng,
// trừ dấu chấm thập phân giữa hai chữ số; "\n" bị escape trong câu trả lời được đổi thành xuống dòng thật.
export function normalizeAIText(text: string): string {
  const source = text.replace(/\r\n?/g, "\n").replace(/\\r\\n|\\n|\\r/g, "\n");
  let result = "";
  let capitalizeNext = true;
  for (let i = 0; i < source.length; i++) {
    const char = source[i];
    if (/\p{L}/u.test(char)) {
      result += capitalizeNext ? char.toLocaleUpperCase("vi-VN") : char;
      capitalizeNext = false;
      continue;
    }
    result += char;
    if (/\p{N}/u.test(char)) {
      capitalizeNext = false;
    } else if (char === "\n") {
      capitalizeNext = true;
    } else if (char === ".") {
      const isDecimalPoint = /\p{N}/u.test(source[i - 1] ?? "") && /\p{N}/u.test(source[i + 1] ?? "");
      if (!isDecimalPoint) capitalizeNext = true;
    }
  }
  return result;
}

// **chữ đậm** thành <strong>; phần còn lại là text node (xuống dòng hiện nhờ white-space: pre-wrap).
export function renderAIText(element: HTMLElement, text: string): void {
  element.replaceChildren();
  const boldPattern = /\*\*([\s\S]+?)\*\*/g;
  let cursor = 0;
  for (const match of text.matchAll(boldPattern)) {
    element.append(text.slice(cursor, match.index));
    const strong = document.createElement("strong");
    strong.textContent = match[1];
    element.append(strong);
    cursor = match.index + match[0].length;
  }
  element.append(text.slice(cursor));
}

export function userMessage(text: string): HTMLElement {
  const row = document.createElement("div");
  row.className = "bubble bubble-user";
  row.textContent = text;
  return row;
}

// Khung câu trả lời như web dev2: tên mô hình (nếu có) ở trên, rồi hàng avatar "SAI" + thân tin nhắn.
function aiRow(label: string | null): { row: HTMLElement; body: HTMLElement } {
  const row = document.createElement("div");
  row.className = "msg-ai";
  if (label) {
    const labelElement = document.createElement("span");
    labelElement.className = "bubble-label";
    labelElement.textContent = label;
    row.append(labelElement);
  }
  const line = document.createElement("div");
  line.className = "msg-ai-row";
  const avatar = document.createElement("span");
  avatar.className = "chat-avatar";
  avatar.setAttribute("aria-hidden", "true");
  avatar.textContent = "SAI";
  const body = document.createElement("div");
  body.className = "msg-ai-body";
  line.append(avatar, body);
  row.append(line);
  return { row, body };
}

// Bong bóng trả lời rỗng kèm nút sao chép; trang chat tự điền chữ (hiện dần hoặc hiện ngay).
// label: tên mô hình hiện trên câu trả lời; null khi lịch sử cũ không lưu mô hình.
export function aiMessage(label: string | null, copyText: () => string): { row: HTMLElement; bubble: HTMLElement } {
  const { row, body } = aiRow(label);
  const bubble = document.createElement("div");
  bubble.className = "bubble bubble-ai glass-control";
  const actions = document.createElement("div");
  actions.className = "msg-actions";
  const copy = iconButton("copy", "Sao chép câu trả lời");
  copy.addEventListener("click", () => copyToClipboard(copy, copyText()));
  actions.append(copy);
  body.append(bubble, actions);
  return { row, bubble };
}

const COPIED_FEEDBACK_MS = 1500;

export async function copyToClipboard(button: HTMLButtonElement, text: string): Promise<void> {
  const label = button.getAttribute("aria-label")!;
  try {
    await navigator.clipboard.writeText(text);
    button.innerHTML = iconSvg("check");
    button.setAttribute("aria-label", "Đã sao chép");
  } catch {
    // Clipboard API chỉ có trong secure context (https, localhost) và có thể bị người dùng chặn.
    button.setAttribute("aria-label", "Không sao chép được");
  }
  setTimeout(() => {
    button.innerHTML = iconSvg("copy");
    button.setAttribute("aria-label", label);
  }, COPIED_FEEDBACK_MS);
}

export function typingMessage(): HTMLElement {
  const { row, body } = aiRow(null);
  body.innerHTML = '<div class="bubble bubble-ai" role="status"><span class="typing-dots"><i></i><i></i><i></i></span><span class="sr-only">SAI đang trả lời</span></div>';
  return row;
}

export function systemMessage(text: string, onRetry?: () => void): HTMLElement {
  const row = document.createElement("div");
  row.className = "msg-system";
  row.setAttribute("role", "alert");
  const span = document.createElement("span");
  span.textContent = text;
  row.append(span);
  if (onRetry) {
    const retry = iconButton("refresh", "Thử lại");
    retry.addEventListener("click", onRetry);
    row.append(retry);
  }
  return row;
}

const SECONDS_PER_DAY = 86400;

function startOfDay(seconds: number): number {
  const date = new Date(seconds * 1000);
  date.setHours(0, 0, 0, 0);
  return date.getTime() / 1000;
}

// Nhóm lịch sử theo ngày (giờ máy người xem).
export function dayGroup(createdAt: number, now: number): string {
  const days = Math.round((startOfDay(now) - startOfDay(createdAt)) / SECONDS_PER_DAY);
  return days <= 0 ? "Hôm nay" : days === 1 ? "Hôm qua" : "Trước đó";
}

export function relativeTime(createdAt: number, now: number): string {
  const minutes = Math.floor((now - createdAt) / 60);
  if (minutes < 1) return "vừa xong";
  if (minutes < 60) return `${minutes} phút trước`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} giờ trước`;
  const days = Math.floor(hours / 24);
  return days === 1 ? "hôm qua" : `${days} ngày trước`;
}

export function groupHeading(text: string): HTMLElement {
  const heading = document.createElement("h4");
  heading.className = "t-overline";
  heading.textContent = text;
  return heading;
}
