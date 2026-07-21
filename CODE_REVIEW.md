# Alpha 项目代码审查报告

- 审查日期:2026-06-07
- 审查范围:`/root/Alpha`,约 18,060 行 Python,26 个源码模块
- 测试基线:**408 passed, 6 subtests passed**(`PYTHONPATH=src python3 -m pytest -q`,本机安装 pytest 9.0.3 后跑通)
- 方法:核心安全模块人工精读 + 6 路并行子代理深度审查大模块,所有 CRITICAL/HIGH 结论已逐条在真实代码上复核(下方标注验证方式)

---

## 总体评价

工程质量整体偏高:测试覆盖完整、分层清晰、提交 guard 是确定性的纯函数。安全基线良好:

- Secrets 处理得当:`.env`、`*.db`、`secrets/**/*.env`、`config/ai_model_profiles.json` 均已 gitignore,仓库内未跟踪任何密钥。
- 全代码库无 `eval` / `exec` / `compile`,无 `shell=True`,无 `verify=False`(TLS 校验从未关闭)。
- 子进程以 argv 列表方式启动,所有用户可控值经白名单校验,**无命令注入**。

### 核心结论:提交 guard 本身不可被 AI 绕过 ✅

已验证关键安全契约成立:

- 候选只有在 `guard.ready`(基于真实 BRAIN 指标的确定性 `evaluate_submission_readiness`)**且** `submit.submitted and submit.stage == "OS"` 时,才会迁移到 `submitted`(`worker.py:954-959`)。
- AI 只产出表达式,从不产出 guard 评估所依据的 metrics/checks,因此 AI 无法伪造 PASS。
- `AUTO_SUBMIT=false` 时调用 `submit_alpha(..., dry_run=True)`,dry-run 分支返回 `submitted=False`,确认无法提交。
- Web 控制台无法启用 auto-submit:`build_daemon_argv` 使用固定 flag 白名单,daemon 子命令中根本不存在 auto-submit flag。

需要修复的问题按严重度列于下。

---

## CRITICAL

### C1. `context_builder.py:497-498` — 未定义名 + `_number` 重复定义,导致默认阈值变成 -999

复现方式:已在本机实际运行 `_targeted_repair_policy` 触发两种后果。

两个 bug 叠加:

1. `DEFAULT_REQUIRED_SHARPE` / `DEFAULT_REQUIRED_FITNESS` 在 `context_builder.py` 中**既未 import 也未定义**(只定义于 `research_planner.py:19-20`)。
2. 模块内有**两个** `def _number`(`context_builder.py:643` 和 `:2304`),后者覆盖前者,坏输入返回 `-999.0`(truthy)而非 `0.0`。

后果:

- `quality_thresholds` 缺失时(常见路径):`_number(None) or DEFAULT_...` 因 `-999.0` 为 truthy 而短路,`required_sharpe` 被悄悄设为 **-999.0**,于是 `sharpe >= required_sharpe` 永远成立,targeted-repair 判断逻辑失效。
- 阈值恰为 `0` 时:`_number` 返回 `0.0`(falsy),触发真正的 `NameError: name 'DEFAULT_REQUIRED_SHARPE' is not defined`,使 optimize cycle 的 context 构建崩溃。

修复:从 `research_planner` import 这两个常量(或本地定义),并删除第二个重复的 `_number`。注意两个 `_number` 返回语义不同(`0.0` vs `-999.0`),合并前需确认各调用点的预期。

---

## HIGH

### H2. daemon 长跑时缺少跨轮 / 平台级提交上限

`cli.py:202-293` vs `submission.py:30-37`

`submitted_this_round` 是 `run_once` 的局部变量,每轮重置为 0。daemon 循环反复调用 `run_once()` 但**从不调用** `submit_approved_candidates`——后者是唯一会查询平台真实提交数(`count_submitted_alphas`)做硬上限的代码。因此 `AUTO_SUBMIT=true` 下,worker 每个 cycle 都可提交至多 `max_final_submits_per_round` 个,无限循环。"每轮 4 个"不等于"每天 4 个"。若设计意图是日级 / 账户级上限,worker 提交路径完全绕过了它。

### H3. approve/submit 路径无异常保护,异常会打死 daemon 并把候选卡在 `approved`

`worker.py:950-956`,`cli.py:201,294`

`_handle_submission_guard` 先 `transition(..., "approved")` 再调 `submit_alpha`,这段无 try/except;daemon 主循环只 catch `KeyboardInterrupt`。提交时一次瞬时 BRAIN 错误或 sqlite 错误会终止整个 daemon,候选永久停在 `approved`(既不 `submitted` 也不 `failed`),且 daemon 不跑 `submit_approved_candidates`,无人重试。

### H4. "trusted" 阈值可把 guard 下限压到配置底线以下

`worker.py:982-988`

`_policy_for_ai_context` 用 `replace(self.policy, **updates)` 直接替换,**而非** `max(self.policy.min_sharpe, value)`。若 trusted thresholds 低于硬编码默认(1.58 / 1.0),有效 guard 会被削弱到配置底线以下。`trusted` 来源为配置 / 观测到的 check limit 而非 AI 原文,故非 AI 直接绕过;但对安全门而言,配置的 `SubmissionPolicy` 最小值不是硬底线是反直觉的。建议加 `max()` 钳制。

### H5. preflight 字段白名单在空列表时 fail-open

`preflight.py:176`

`if allowed_fields:` 意味着调用方传空 / None 白名单时,整个 `UNKNOWN_FIELD` 检查被跳过,任意幻觉字段名都能过门。可达路径:`worker._allowed_fields_from_context` 与 `clients._context_allowed_fields` 在 research_context 缺失 / 畸形时均返回 `[]`。非代码执行风险(无 eval),而是安全门 fail-open,会让畸形表达式进 BRAIN 烧仿真配额。建议区分"未提供白名单"与"按白名单校验"两种语义。

### H6. 硬模式机制样本仍原样泄露被禁表达式

`context_builder.py:1835, 478`;`research_planner.py:2123-2136`

设计声明硬模式只传"机制"、不泄露 forbidden fields/expression。但每个 archetype 在挂上 `"policy": "Do not copy this expression..."` 的同时,**仍原样带上完整 `expression` 字段**(其中即含 forbidden fields)。"不要泄露"的保证只是一句礼貌请求摆在被泄露数据旁边。若意图为机制-only,序列化前应删除 `expression`(或做结构抽象)。

### H7. (Web,单用户场景权重降低)默认绑 0.0.0.0 + 无认证 + 状态变更端点无 CSRF

`web.py:585, 641-667`

`/api/start`、`/api/stop`、`/api/clear-logs` 无任何认证与 CSRF/Origin 校验,`_read_json` 不校验 Content-Type。用户浏览恶意页面时可被 CORS simple request 触发(停 daemon、清空全部日志、起 daemon)。建议默认绑 `127.0.0.1`,加共享 token,POST 加 Origin/Referer 校验。

---

## MEDIUM

- **超时线程泄露 + 重试派生并发真实仿真**(`worker.py:532-568, 1423`;`clients.py:1814-1824`):`future.cancel()` 停不掉已运行任务,底层 HTTP 仍在跑;`_simulate_candidates` 对 TimeoutError 重试至多 3 次,一个真卡住的仿真会派生至多 3 个并发孤儿线程,各自发真实 BRAIN 仿真,浪费配额并可能产生重复 alpha。
- **`db.transition()` 非原子**(`db.py:300-302`):`update_candidate` 与 `record_event` 各开各的连接 / 事务,中途崩溃会出现"状态改了没事件"或"有事件没改状态"。最核心的状态迁移原语反而最不原子(对比 `archive_candidates` 单事务)。
- **BRAIN 会话过期不重登录 + `Retry-After` 裸 `float()`**(`clients.py:3329-3346, 3433`):长跑时 session 过期后所有调用永久 401 无恢复;`Retry-After` 若为 HTTP-date 形式,`float()` 抛 `ValueError` 直接中断仿真(作者在 `_get_with_rate_limit_retry` 里已 catch,说明知情,但未统一应用)。
- **per-candidate 全表扫描**(`worker.py:618` `_find_structural_duplicate`;`context_builder.py:1517+` history_memory):每候选 `list_candidates()` 全表 + 每候选 3-5 次 events 查询 × 300,每轮约 1000-1500 次往返,为每轮主要开销且未缓存。
- **`field_stats` 无上限序列化进 AI context**(`context_builder.py:1612`):以 `len(field_stats)` 作上限等于不设限,长跑 scope 会持续膨胀 token 预算(其他 bucket 均封顶 20/30)。
- **daemon 子进程从不回收**(`web.py:222-231`):`Popen` 为局部变量,从不 `wait()`/`poll()`,自行退出的 daemon 变僵尸进程累积。
- **`stop()` 只发主 PID**(`web.py:256-295`):用 `start_new_session=True` 建了进程组却只对 `pid` 发信号(非 `-pid`);无 SIGKILL 兜底;存在 PID 复用误杀风险。
- **scheduler 大量裸 `except Exception` 返回空**(`scheduler.py` 多处):把 schema drift 等真 bug 掩盖成"无信号",静默降级调度决策。
- **structure-diversity 回退分支误把成功骨架计为失败**(`research_planner.py` `_structure_diversity_control` 回退分支):`failure_rate=1.0` 不分状态,downstream 会抑制真正成功的骨架。

---

## LOW

- "最近 approved 字段须避开"仅为软提示,无硬拒绝(对比辅助-主字段规则有真实 preflight 门)。
- `db.py:61-64`:`timeout=30` 被 `PRAGMA busy_timeout=5000` 覆盖为 5s;FK 声明未生效(未开 `PRAGMA foreign_keys=ON`);无 schema 迁移 / 版本机制。
- `clients.py`:429 / 瞬时错误处理不一致(仅 `discover_datafields`/`get_alpha_detail` 有重试),`requests.Session` 从未关闭。
- `_lit_tower_avoidance` 的 `examples` 永远为空 `[]`(`context_builder.py:953, 996`);存在多个死函数(`_extract_authoritative_pyramid_towers`、`_candidate_probe_datasets`、`_with_dataset_risk_fallback` 等)。
- Web POST 返回 `str(exc)` 泄露内部路径;`_read_json` 无 body 大小上限。
- `ModelProfile` 默认 `repr` 含 `api_key`(`clients.py:68-76`),当前无日志打印,但属潜在泄露点,建议 `field(repr=False)`。
- preflight `EXACT_OPERATOR_ARITY` 对带可选 / 命名参数的合法表达式可能误拒(fail-closed,非绕过)。

---

## 建议修复优先级

1. **先修 C1**:一行 import + 删除重复 `_number`。这是会真崩溃且静默改变阈值语义的硬 bug。
2. **明确 H2 意图**:若需日级上限,daemon 路径需接入 `count_submitted_alphas`;给 H3 加 `except Exception`,让候选落到 `failed` 而非卡死。
3. **收紧安全门**:H4 加 `max()` 钳制、H5 区分空白名单语义、H6 序列化前删 `expression`——三者均为"安全门未按声明那样严格"的缺口。
4. **Web 加固**:H7 默认绑 loopback + token + Origin 校验。
5. 其余 MEDIUM / LOW 视运维优先级排期。

---

## 已修复(2026-06-07)

本次会话已修复并通过全套测试(411 passed):

- **C1 已修复** — `context_builder.py`:从 `research_planner` import `DEFAULT_REQUIRED_SHARPE`/`DEFAULT_REQUIRED_FITNESS`;删除第 2304 行返回 `-999.0` 的重复 `_number` 定义,保留返回 `0.0` 的版本。崩溃路径与 -999 静默默认均已消除。
- **H4 已修复** — `worker.py:_policy_for_ai_context`:trusted 阈值改为只能收紧 guard。`min_*` 用 `max(current, value)` 上钳,`max_turnover` 用 `min(current, value)` 下钳,配置 `SubmissionPolicy` 底线成为硬下限。新增测试 `test_policy_for_ai_context_does_not_relax_below_configured_floor`。
- **H5 已修复** — preflight 与 worker 双层处理:
  - `preflight.validate_expression`:`allowed_fields=None` 表示"未提供白名单(跳过)",显式传入的可迭代对象(含空列表)按白名单校验(空 = 全拒,fail-closed)。新增两条 preflight 测试。
  - `worker.run_once`:在 AI 自由生成分支前加熔断,字段池为空时记 `empty_field_pool_generation_blocked` 事件并跳过本轮(不再逐个生成逐个拒)。新增测试 `test_worker_skips_fresh_ai_generation_when_field_pool_is_empty`。
  - probe/planned 与 `MultiModelAIClient._local_preflight_errors` 的合法空池路径改为传 `... or None`,保留跳过语义。
- **H2 已修复** — `worker.run_once`:新增 `_platform_submitted_today()`,在 `AUTO_SUBMIT=true` 下用 `count_submitted_alphas` + `trading_day_window` 把 `submitted_this_round` 种子化为平台当日真实提交数,使每轮提交上限成为真正的日级/账户级硬约束(不再每 cycle 重置)。平台计数不可用时按"已达上限"处理,拒绝继续提交。新增两条 `test_submission_flow` 测试。
- **H3 已修复** — `worker._handle_submission_guard` 的 `submit_alpha` 加 try/except:提交异常时候选保持 `approved`(供 `submit-approved` 重试)并记 `submit_error` 事件,不再让瞬时错误冒泡。`cli.py` daemon 循环把每轮工作体包进 `except Exception`,记 `daemon_cycle_error` 事件、退避、继续,单轮失败不再打死 daemon。
- **H6 已修复** — 硬模式机制样本不再泄露被禁表达式:`context_builder._blocked_winner_archetype`、`_submitted_target_mechanism_transfer`、`research_planner._mechanism_memory_from_history` 三处删除 `expression` 字段,只保留 `mechanism_tags`/`forbidden_fields`/`transfer_hint`/families。新增断言验证 archetype 不含 `expression`。
- **H7 已修复** — Web 加固:`run_web_app`/cli/`scripts/alpha_web` 默认绑 `127.0.0.1`;新增 `ALPHA_WEB_TOKEN` 鉴权(`X-Alpha-Token` 头或 `?token=`,`hmac.compare_digest` 常量时间比较);状态变更 POST 加同源(Origin/Referer)校验;`_read_json` 加 1 MiB body 上限;非回环且无 token 时启动打印警告。README/.env.example 已补文档。新增 4 条 `WebSecurityTests`(端到端起服务验证 token/origin/body-cap)。

### MEDIUM(本次已修复)

- **`db.transition` 原子化** — UPDATE 状态与 INSERT 事件合并到单连接单事务,杜绝"改了状态没事件"或反之。
- **SQLite PRAGMA** — `connect()` 启用 `foreign_keys=ON`;`busy_timeout` 从 5000 修正为 30000(与 `timeout=30` 一致)。测试断言已更新。
- **`Retry-After` 安全解析** — `clients.py` 新增 `_parse_retry_after()`,支持 delay-seconds 与 HTTP-date 两种形式,缺失/不可解析回退默认值并钳制到上限;替换全部 6 处裸 `float(retry_after)`。新增 `RetryAfterTests`。
- **Web 子进程管理** — `stop()` 升级为 SIGINT → SIGTERM → SIGKILL 升级链,并经 `_signal_process` 优先对进程组(`killpg`)发信号、回退单 PID;新增 `_reap_daemon_process()` 在 `status()`/`stop()` 回收已退出 daemon,避免僵尸。
- **scheduler 裸 except** — 6 处 `try: list(reader(...)) except Exception: return <default>` 收敛为单一 `_read_events()` 辅助函数,读失败按 debug 级日志记录而非完全静默。
- **structure-diversity 回退** — 同轮重复骨架不再伪造 `failed`/`failure_rate=1.0`,改为如实标注 `same_round_repeats`/`reason="same_round_repetition"`。
- **仿真 stage 超时不再重试** — `_simulate_one_with_timeout` 超时返回 `SimulationFailure("simulation_stage_timeout")` 而非 raise,避免重试派生重复真实仿真、浪费配额。新增 `test_worker_does_not_retry_simulation_stage_timeout`。

全套测试:**423 passed, 6 subtests passed**。

待办(本次未改,LOW):软性"最近 approved 字段避让"无硬拒、无 schema 迁移机制、`requests.Session` 未关闭、`_lit_tower_avoidance.examples` 恒空与死函数清理、`ModelProfile` repr 含 key、preflight 精确 arity 误拒、find_duplicate/structural-dup 全表扫描的性能优化。
