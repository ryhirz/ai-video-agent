/* 前端阶段2 验收验证（node，无浏览器）：
 * 1) 以 DOM 桩真实加载 editor.html 的 <script>；
 * 2) 正例：speed+grayscale+add_text 组合 → validate 通过 + buildTimeline 形态正确；
 * 3) 反例：transform.x 负偏移 / speed 越界 → 前端拦截且提示可读；
 * 4) 正例 timeline 写盘供 python 侧真渲染。
 * 运行：node tests/editor_verify.mjs
 */
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const html = readFileSync(join(root, "server/static/editor.html"), "utf-8");
const js = html.match(/<script>([\s\S]*?)<\/script>/)[1];

/* ---- DOM 桩：够编辑器顶层执行 + 交互函数调用即可 ---- */
function makeEl() {
  return {
    innerHTML: "", value: "", textContent: "", style: {}, checked: false,
    className: "", children: [],
    appendChild(c) { this.children.push(c); },
    querySelectorAll() { return []; },
    querySelector() { return null; },
    closest() { return null; },
    setAttribute() {}, getAttribute() { return null; },
    onclick: null, onchange: null,
  };
}
const elCache = {};
globalThis.document = {
  createElement: () => makeEl(),
  getElementById: (id) => (elCache[id] ??= makeEl()),
};
const inputs = {};                        // 模拟用户输入框的值
globalThis.fetch = () => new Promise(() => {});   // manifest 拉取挂起 → 走兜底
/* 浏览器 API 桩：编辑器顶层初始化会读 localStorage（API Key 记忆） */
globalThis.localStorage = {
  _s: {}, getItem(k) { return this._s[k] ?? null; },
  setItem(k, v) { this._s[k] = String(v); },
  removeItem(k) { delete this._s[k]; },
};
globalThis.alert = () => {};

const window = globalThis;
/* ESM 严格模式下 eval 不泄漏 var —— 用 new Function 包裹并显式引出 API */
const api = new Function(
  "document", "fetch", "window",
  js + "\n;return { state, validate, buildTimeline, renderAll, applyDrag };")(
  globalThis.document, globalThis.fetch, window);
const { state, validate, buildTimeline, applyDrag } = api;

/* 注入 manifest 兜底开关：强制走 FALLBACK（fetch 已挂起） */
// 直接操纵 state（模拟用户在面板上的编辑结果）
state.clips = [
  { id: "v1", track: "v_main", sourceRef: "E:/media/sample.mp4",
    "in": 0, out: 4, sourceIn: 0, sourceOut: 4,
    transform: { x: 0, y: 0, scale: 1.0, rotation: 0, opacity: 1.0, crop: null },
    effects: [{ effectType: "speed", params: { rate: 1.5 } },
              { effectType: "grayscale", params: {} }] },
  { id: "a1", track: "a_main", sourceRef: "E:/media/sample.mp4",
    "in": 0, out: 4, sourceIn: 0, sourceOut: 4,
    transform: { x: 0, y: 0, scale: 1.0, rotation: 0, opacity: 1.0, crop: null },
    effects: [{ effectType: "speed", params: { rate: 1.5 } }] },
];
state.texts = [{ id: "tx1", text: "水印TEST", start: 0, end: 2,
  position: "top-right", margin: 20, size: 28, color: "#FFFFFF" }];

/* ---- 正例：validate 通过 + buildTimeline 形态 ---- */
let errs = validate();
if (errs.length) { console.error("正例误报:", errs); process.exit(1); }
const tl = buildTimeline();
const cv = tl.clips[0];
const okFx = cv.effects.some(e => e.effectType === "speed" && e.params.rate === 1.5)
          && cv.effects.some(e => e.effectType === "grayscale");
if (!okFx || tl.texts.length !== 1 || tl.duration !== 4
    || cv.transform.x !== 0 || tl.schemaVersion !== "1.0.0") {
  console.error("buildTimeline 形态不符:", JSON.stringify(tl, null, 1));
  process.exit(1);
}
console.log("PASS 正例：validate 通过，组合 timeline 形态正确");

/* ---- 反例 1：transform.x 负偏移 → 前端拦截 ---- */
state.clips[0].transform.x = -5;
errs = validate();
if (!errs.some(e => e.includes("transform.x 不可为负"))) {
  console.error("负偏移未被拦截:", errs); process.exit(1);
}
console.log("PASS 反例1：transform.x=-5 被前端拦截 →", errs[0].trim());

/* ---- 反例 2：speed rate 越界 → 前端拦截 ---- */
state.clips[0].transform.x = 0;
state.clips[0].effects[0].params.rate = 2000;
errs = validate();
if (!errs.some(e => e.includes("speed rate"))) {
  console.error("speed 越界未被拦截:", errs); process.exit(1);
}
console.log("PASS 反例2：rate=2000 被前端拦截 →",
            errs.find(e => e.includes("speed rate")).trim());

/* ---- 还原正例并写盘供 python 真渲染 ---- */
state.clips[0].effects[0].params.rate = 1.5;
const final = buildTimeline();
writeFileSync(join(root, "server/server_data/_editor_combo.json"),
              JSON.stringify(final, null, 1));
console.log("组合 timeline 已写盘：server/server_data/_editor_combo.json");

/* ---- 拖拽修剪纯逻辑 applyDrag ---- */
const dragClip = { id: "d1", track: "v_main", sourceRef: "E:/m.mp4",
  "in": 2, out: 5, sourceIn: 1, sourceOut: 4, effects: [] };
// 左缘 +0.5s：入点与素材入点同步后移（原速 rate=1）
let o = applyDrag(dragClip, "L", 0.5);
if (!o || o["in"] !== 2.5 || o.sourceIn !== 1.5 || o.out !== 5
    || o.sourceOut !== 4) {
  console.error("applyDrag L 正常修剪不符:", o); process.exit(1);
}
// 左缘越界：dt=-3 → in=-1 拒绝；素材耗尽（sourceIn<0）拒绝
if (applyDrag(dragClip, "L", -3) !== null
    || applyDrag(dragClip, "L", -1.5) !== null) {
  console.error("applyDrag L 越界未拒绝"); process.exit(1);
}
// 右缘 -1s：out/sourceOut 同步收缩
o = applyDrag(dragClip, "R", -1);
if (!o || o.out !== 4 || o.sourceOut !== 3 || o["in"] !== 2) {
  console.error("applyDrag R 收缩不符:", o); process.exit(1);
}
// speed=2 时素材变化 = 时间轴变化 × rate
const fast = { ...dragClip, effects: [{ effectType: "speed",
                                       params: { rate: 2 } }] };
o = applyDrag(fast, "R", 0.5);
if (!o || o.out !== 5.5 || o.sourceOut !== 5) {   // 4 + 0.5*2
  console.error("applyDrag speed 换算不符:", o); process.exit(1);
}
console.log("PASS 拖拽：applyDrag 边界约束 + speed 换算正确");

/* ---- 多轨：新增视频轨 + 片段归属 + 轨道不存在拦截 ---- */
state.tracks.push({ id: "v_2", type: "video" });
state.clips.push(
  { id: "v2a", track: "v_2", sourceRef: "E:/media/sample.mp4",
    "in": 4, out: 6, sourceIn: 0, sourceOut: 2,
    transform: { x: 0, y: 0, scale: 1.0, rotation: 0, opacity: 1.0, crop: null },
    effects: [] });
errs = validate();
if (errs.length) { console.error("多轨正例误报:", errs); process.exit(1); }
const multi = buildTimeline();
if (multi.tracks.length !== 3
    || !multi.tracks.some(t => t.id === "v_2" && t.type === "video")
    || !multi.clips.some(c => c.track === "v_2")) {
  console.error("多轨 buildTimeline 不符:",
                JSON.stringify(multi.tracks), JSON.stringify(multi.clips));
  process.exit(1);
}
console.log("PASS 多轨：3 轨 timeline 组装正确（视频轨按入点顺序合并语义）");

// 轨道不存在 → 前端拦截
state.clips[0].track = "v_gone";
errs = validate();
if (!errs.some(e => e.includes("轨道 v_gone 不存在"))) {
  console.error("孤儿轨道未被拦截:", errs); process.exit(1);
}
state.clips[0].track = "v_main";
console.log("PASS 多轨：片段引用不存在轨道被拦截");
