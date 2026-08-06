# SwiftAgentX 优化计划（每日一项）

> 来源：2026-08-04 对当前 AI 热点（deepagents / graph 工程 / loop 工程）的整合分析。
> 原则：只放大 Scenario 核心差异化，不做图 DSL、不做 deepagents 式全套 harness 仿品。
>
> **执行规则**
> - 分支：`daily-opt`，每天完成一项、勾选对应 checkbox、commit 并 push（push 前 `git pull --rebase origin daily-opt`）。
> - 每项先写测试（TDD），完成标准 = 该项"验证方式"通过。
> - 涉及 README 的改动必须中英文段同一 commit 同步。
> - 不保留兼容层：直接改干净，不留 deprecated 路径。

## 进度

- [x] **D1 · Scenario 并行步骤组（结构 + 执行器）**（完成于 2026-08-04）
  `ScenarioConfig.tool_chain` 支持并行组（一组无依赖 step fan-out 后 join，`asyncio.gather`），数据结构保持 list-of-lists 级别的简单扩展，不引入图抽象。
  验证：单测（并行结果合并、变量注入）+ benchmark 对比同链串行 vs 并行的 P50。
  benchmark 待本地补测（云端环境无法跑真实 LLM/网络延迟对比）。

- [x] **D2 · 并行组失败语义与条件边**（完成于 2026-08-05）
  并行组的部分失败策略（`fail_fast` / `best_effort`），step `condition` 在 join 后统一求值；失败回退到 ReAct 的现有路径保持不变。
  验证：单测覆盖全部失败/条件分支组合。

- [x] **D3 · Tool 大结果 offload（context 卸载）**（完成于 2026-08-06）
  超过阈值的 tool 输出写入 workspace 文件，context 中只保留引用 + 摘要；ReAct 和 Scenario hook 可按需回读。
  验证：单测 + 长输出场景 token 用量前后对比。
  token 用量前后对比 benchmark 待本地补测（云端环境无法跑真实 LLM 调用）。
  对抗式复审后补修（2026-08-06）：读回闭环（workspace_read 结果不再被二次卸载，
  改为分块 + offset 分页）、planner 快路径同样过卸载、offload key 加随机后缀防跨轮
  覆盖、workspace 写失败降级为截断内联而非请求失败。

- [ ] **D3b · direct 大结果的 memory 回灌卸载**
  `output_type="direct"` 的大结果作为 answer 进 L2 verbatim memory 后，会在之后每轮
  被整段注回 prompt，跨轮维度绕过了 D3。需在 `add_turn` 入 L2 前对超阈值 answer 做同样
  的 preview + 引用处理（答案本身仍原样返回用户）。
  验证：单测（第二轮 prompt 不含第一轮完整大结果，但用户收到的答案完整）。

- [ ] **D4 · 链执行状态持久化（checkpoint / 恢复）**
  长链每步执行状态落 storage，失败或人工审批中断后可从断点恢复；与现有 promotion 人工门衔接。
  验证：中断-恢复单测（模拟进程退出后 resume）。

- [ ] **D5 · 挖矿 loop 一期：transcript 采集与模式聚类**
  后台任务扫描历史请求记录，聚类重复出现的 ReAct 工具序列，产出 Scenario 候选（进入现有候选池）。
  验证：用 benchmarks 流量重放，确认能自动产出预期候选。

- [ ] **D6 · 回放评测门（replay eval gate）**
  候选链用历史真实请求 replay，输出与 ReAct 基线做一致性打分；达标才进入审批队列，报告落盘。
  验证：eval 报告生成 + 阈值判定单测。

- [ ] **D7 · 自动转正 gate**
  规则化自动 promote：连续 N 次成功 + eval 达标 + 显式开启开关时跳过人工审批；默认关闭。
  验证：端到端单测（候选 → eval → 自动转正 → Scenario 命中）。

- [ ] **D8 · Agent Skills（SKILL.md）标准 loader**
  读取 Anthropic Agent Skills 目录格式（SKILL.md + 资源），映射到现有 Skill-in-ReAct 机制，生态技能包直接可用。
  验证：加载真实 SKILL.md 样例并在 ReAct 中触发执行。

- [ ] **D9 · 文档与 benchmark 收尾**
  README（中英同步）、CHANGELOG、architecture 文档更新；重跑 benchmark 更新数据与图。
  验证：`pip install -e ".[dev]"` 全量测试通过 + benchmark 图表更新。

## 明确不做（红线）

- 图 DSL / 状态机 API —— 破坏"一个下午读完"的定位。
- deepagents 式全套 harness（filesystem todos + CLI）—— 稀释 Scenario 差异化。
- 多 agent 协作协议（A2A 等）—— 与单 agent 低延迟定位不符。
