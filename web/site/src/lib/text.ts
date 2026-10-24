// Chuyển văn bản lấy từ docstring Python sang HTML an toàn: ``x`` (cú pháp RST) thành <code>x</code>.
const ESCAPES: Record<string, string> = { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" };

function escapeHtml(text: string): string {
  return text.replace(/[&<>"]/g, char => ESCAPES[char]);
}

export function docstringToHtml(text: string): string {
  return escapeHtml(text).replace(/``([^`]+)``/g, "<code>$1</code>");
}

// Docstring viết chữ thường đầu câu sau dấu ":"; trên web là câu đứng riêng nên viết hoa chữ đầu.
export function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

// Số tham số theo triệu, viết kiểu Việt Nam: 38683136 → "38,7".
export function formatMillions(count: number): string {
  return (count / 1e6).toLocaleString("vi-VN", { maximumFractionDigits: 1 });
}
