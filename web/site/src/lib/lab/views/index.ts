// Lệnh lab → phần riêng của lệnh đó trong phòng làm việc. Thêm phòng mới: viết view rồi thêm một dòng ở đây và một
// trang trong ../workbench.ts.
import { kernelView } from "./compareKernel";
import { objectiveView } from "./compareObjective";
import { inferView } from "./infer";
import type { TaskView } from "./view";

export const TASK_VIEWS: Record<string, TaskView> = {
  "compare-kernel": kernelView,
  "compare-objective": objectiveView,
  infer: inferView,
};
