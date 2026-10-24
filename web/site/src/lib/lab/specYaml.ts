// Viết một phần spec thành YAML theo cách viết của file spec trong lab (vd. foundation_lab/inference/configs/
// compare_35M.yaml): bỏ khoá bằng mặc định trong schema; object chỉ chứa giá trị đơn viết gọn trên một dòng.
import type { ObjectSchema } from "./labApi";

// Chuỗi viết trần được: bắt đầu bằng chữ hoặc "_", không có ký tự mà YAML hiểu theo nghĩa khác (":", "#", ngoặc...).
const PLAIN_STRING = /^[\p{L}_][\p{L}\p{N}_.\-/ ]*$/u;
// Chữ mà YAML đọc thành bool/null nếu viết trần.
const YAML_KEYWORDS = new Set(["true", "false", "yes", "no", "on", "off", "null"]);
const INDENT = "  ";

const isObject = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

// Giá trị khác mặc định của schema; object lồng chỉ giữ khi còn khoá khác mặc định.
export function withoutDefaults(value: Record<string, unknown>, schema: ObjectSchema): Record<string, unknown> {
  const result: Record<string, unknown> = {};
  for (const [name, item] of Object.entries(value)) {
    const field = schema.fields[name];
    if (field.type === "object" && isObject(item)) {
      const nested = withoutDefaults(item, field);
      if (Object.keys(nested).length > 0) result[name] = nested;
    } else if (!("default" in field) || JSON.stringify(item) !== JSON.stringify(field.default)) {
      result[name] = item;
    }
  }
  return result;
}

function scalar(value: unknown): string {
  if (typeof value === "string" && PLAIN_STRING.test(value) && value.trim() === value && !YAML_KEYWORDS.has(value.toLowerCase())) {
    return value;
  }
  // Chuỗi JSON là chuỗi YAML hợp lệ trong ngoặc kép; số, bool, null viết như JSON cũng đúng YAML.
  return JSON.stringify(value);
}

function flow(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(flow).join(", ")}]`;
  if (isObject(value)) return `{${Object.entries(value).map(([key, item]) => `${key}: ${flow(item)}`).join(", ")}}`;
  return scalar(value);
}

function block(value: Record<string, unknown>, indent: string): string[] {
  return Object.entries(value).flatMap(([key, item]) =>
    isObject(item) && Object.values(item).some(isObject)
      ? [`${indent}${key}:`, ...block(item, indent + INDENT)]
      : [`${indent}${key}: ${flow(item)}`],
  );
}

// Một phần tử của danh sách (vd. một biến thể dưới "variants:"): "- " ở dòng đầu, các dòng sau thụt vào.
export function listItemYaml(value: Record<string, unknown>): string {
  return block(value, "").map((line, index) => (index === 0 ? "- " : INDENT) + line).join("\n");
}
