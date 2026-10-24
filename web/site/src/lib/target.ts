// Bản deploy đang build. astro.config.mjs đã kiểm tra giá trị; ở đây kiểm tra lại để có kiểu chặt cho trang.
const TARGETS = ["static", "server"] as const;
export type DeployTarget = (typeof TARGETS)[number];

export function deployTarget(): DeployTarget {
  const value: string = import.meta.env.PUBLIC_DEPLOY_TARGET;
  if (!(TARGETS as readonly string[]).includes(value)) {
    throw new Error(`PUBLIC_DEPLOY_TARGET phải là "static" hoặc "server", nhận được: ${value}`);
  }
  return value as DeployTarget;
}
