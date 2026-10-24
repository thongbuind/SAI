// Icon của từng phòng ban trong lab.json. Chỉ là trình bày nên đặt ở web, không đưa vào dữ liệu xuất từ lab.
import type { IconName } from "./icons";

const DEPARTMENT_ICON: Record<string, IconName> = {
  core: "sliders", arch: "layers", data: "database", tokenizer: "type", objectives: "target",
  engine: "cpu", state: "hard-drive", recipes: "flask", eval: "chart", inference: "bolt",
};

export function departmentIcon(code: string): IconName {
  const icon = DEPARTMENT_ICON[code];
  if (!icon) {
    throw new Error(`Phòng "${code}" chưa có icon trong src/lib/departments.ts`);
  }
  return icon;
}
