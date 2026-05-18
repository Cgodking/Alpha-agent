# Alpha Agent

WorldQuant BRAIN Alpha 自动探索、回测、筛选与提交代理。

这个项目用于在个人服务器上常驻运行一个受控的 Alpha 研究闭环：AI 生成候选表达式，服务端做本地预检，调用 BRAIN 回测，读取真实指标和提交检查，筛选达标候选，并在满足规则后自动提交。AI 不能直接提交，所有提交都必须经过服务端 guard。

核心流程：

```text
generated -> preflight_passed -> simulated -> metric_passed
-> check_pending | approved -> submitted
```

## 安全默认值

- 默认 `AUTO_SUBMIT=false`，不会自动提交。
- 每轮最多最终提交 `4` 个 Alpha。
- 所有强制检查必须最终 `PASS` 才能提交。
- `PENDING`、`FAIL`、`ERROR` 和强制项 `WARNING` 都会阻断提交。
- self/prod correlation 超过 `0.7` 会阻断提交。
- AI 表达式会先经过本地算子、字段和结构预检，再进入 BRAIN 回测。
- 每次生成前都会给 AI 提供研究上下文：本地规则、生成经验、候选历史、字段池、失败原因、近似达标候选和旧 Brain 项目参考信息。
- 本地 `local` 模式只用于测试服务逻辑，不会调用真实 BRAIN。

## 常用命令

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db init-db
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db status
PYTHONPATH=src python3 -m alpha.cli --db alpha.db fields --preset ind --limit 40
PYTHONPATH=src python3 -m alpha.cli --db alpha.db submit-approved
PYTHONPATH=src python3 -m alpha.cli --db alpha.db daemon --batch-size 8 --loop-seconds 60
PYTHONPATH=src python3 -m alpha.cli --db alpha.db web --host 0.0.0.0 --port 8080
```

CLI 默认读取 `.env`，日志写入 `logs/alpha.log`。可以用 `--env-file` 和 `--log-file` 覆盖。

不改 `.env` 的情况下指定探索范围：

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset us-d0 --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset chn-d0 --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset eur-d1 --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset glb --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --preset ind --batch-size 1
PYTHONPATH=src python3 -m alpha.cli --db alpha.db run-once --region USA --universe TOP3000 --delay 0 --neutralization SUBINDUSTRY --decay 6 --truncation 0.03 --batch-size 1
```

## AI 配置示例

复制 `.env.example` 到 `.env`，再填入本地密钥。

共享一个中转 token 的示例：

```bash
AI_CLIENT=multi
AI_API_KEY=your_key_here
AI_BASE_URL=https://your-relay-host/v1
AI_MODEL_PROFILES=grok-4-1-fast-reasoning@controller,deepseek-v4-pro-free@critic,gemini-3-flash-free@generator,glm-4.5@optimizer,gpt-5.4-nano@validator
BRAIN_CLIENT=local
AUTO_SUBMIT=false
```

每个模型使用不同 token 时，建议使用 profile 文件：

```bash
AI_CLIENT=multi
AI_MODEL_PROFILES_FILE=config/ai_model_profiles.json
BRAIN_CLIENT=http
AUTO_SUBMIT=false
```

从示例文件创建本地配置：

```bash
cp config/ai_model_profiles.example.json config/ai_model_profiles.json
```

每个 `secrets/ai/*.env` 文件可以放一个模型的中转 token 和地址：

```bash
AI_API_KEY=one_model_token_here
AI_BASE_URL=https://api.1314mc.net/v1
```

`config/ai_model_profiles.json`、`secrets/**/*.env`、`.env` 都已被 git 忽略。示例文件只放占位符，可以安全提交。

## BRAIN 配置

真实 BRAIN dry-run 回测：

```bash
BRAIN_CLIENT=http
BRAIN_EMAIL=your_brain_email
BRAIN_PASSWORD=your_brain_password
AUTO_SUBMIT=false
```

如果已有旧 Brain 项目的凭据文件，也可以使用：

```bash
BRAIN_CLIENT=http
BRAIN_CREDENTIALS_FILE=/root/brain_alpha/Brain/brain_credentials.txt
AUTO_SUBMIT=false
```

`BRAIN_CLIENT=local` 不会真实回测，只用于本地测试和联调。

## 前端控制台

启动个人控制台：

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db web --host 0.0.0.0 --port 8080
```

打开：

```text
http://SERVER_IP:8080
```

服务器常驻版默认使用 `alpha-web.service`，端口 `5000`：

```bash
scripts/alpha_web start
scripts/alpha_web status
scripts/alpha_web restart
scripts/alpha_web stop
```

控制台可以：

- 选择 `region`、`delay`、`universe`、`neutralization`
- 使用预设快速切换常见 scope
- 后台启动或暂停 daemon
- 设置单轮 batch size，默认 `8`，不超过 BRAIN multisimulation 上限
- 查看候选状态、最近表达式、模型表现和当前研究计划
- 查看字段池
- 查看并清空日志

控制台启动的 daemon 仍然走同一套 CLI guard：

```bash
PYTHONPATH=src python3 -m alpha.cli --db alpha.db daemon --region <region> --universe <universe> --delay <delay> --neutralization <neutralization> --batch-size 8 --loop-seconds <seconds>
```

从前端启动或停止不会绕过字段验证、BRAIN 回测、重试上限、相关性检查或最终提交上限。

## 关键环境变量

- `AUTO_SUBMIT`: 默认 `false`
- `AI_CLIENT`: 默认 `local`；`openai` 表示单模型 OpenAI-compatible API；`multi` 表示多模型编排
- `AI_API_KEY` / `OPENAI_API_KEY`: AI API token
- `AI_BASE_URL`: OpenAI-compatible API 地址，可以是 `https://host/v1` 或 `https://host/v1/chat/completions`
- `AI_MODEL`: 单模型模式使用的模型名
- `AI_MODEL_PROFILES`: 多模型紧凑配置
- `AI_MODEL_PROFILES_FILE`: 多模型 JSON 配置文件，优先级高于 `AI_MODEL_PROFILES`
- `BRAIN_CLIENT`: 默认 `local`；真实平台使用 `http`
- `BRAIN_EMAIL` / `BRAIN_PASSWORD`: BRAIN 登录凭据
- `BRAIN_CREDENTIALS_FILE`: 兼容旧 Brain 项目的凭据文件
- `BRAIN_BASE_URL`: 默认 `https://api.worldquantbrain.com`
- `ALPHA_REGION`: 默认 `USA`
- `ALPHA_UNIVERSE`: 默认 `TOP3000`
- `ALPHA_DELAY`: 默认 `1`
- `ALPHA_NEUTRALIZATION`: 默认 `INDUSTRY`
- `ALPHA_DECAY`: 默认 `0`
- `ALPHA_TRUNCATION`: 默认 `0.05`
- `ALPHA_FIELD_DISCOVERY`: 默认 `true`
- `ALPHA_FIELD_LIMIT`: 默认 `120`
- `ALPHA_FIELD_SEARCHES`: 字段搜索词，逗号分隔
- `ALPHA_FIELD_CACHE_DIR`: 默认 `data/field_cache`
- `ALPHA_FIELD_CACHE_TTL_SECONDS`: 默认 `86400`
- `ALPHA_KNOWLEDGE_DIR`: 默认 `knowledge/`
- `REFERENCE_BRAIN_DIR`: 默认 `/root/brain_alpha/Brain`
- `MAX_FINAL_SUBMITS_PER_ROUND`: 默认 `4`
- `MAX_RETRIES`: 默认 `3`
- `MIN_SHARPE`: 默认 `1.58`
- `MIN_FITNESS`: 默认 `1.0`
- `MAX_CORRELATION`: 默认 `0.7`
- `ALPHA_DB`: 默认 `alpha.db`
- `BATCH_SIZE`: 默认 `8`
- `LOOP_SECONDS`: 默认 `60`

## AI 研究上下文

每轮生成前，worker 会构建 `research_context`，主要包括：

- `knowledge/wqb_rules.md`
- `knowledge/generation_patterns.md`
- `knowledge/optimization_playbook.md`
- `knowledge/usa_d0_patterns.md`
- 当前 scope 的 BRAIN datafields 字段池
- 最近失败、pending、approved、submitted 候选
- 候选分层队列：`submitable`、`watchlist`、`optimize`、`trash`、`abandoned`
- 历史字段和字段家族表现
- 已提交字段避让信息
- 当前季度已点亮塔避让信息
- 旧 Brain 项目的提交、失败和模板摘要

真实回测完成后，系统会把 alpha id、IS 指标、submission checks、AI hypothesis、risk notes、失败原因和状态迁移写入 SQLite。下一轮 AI 会看到这些反馈，从而减少重复错误，并围绕真实平台结果做改进。

## 多模型协作

`AI_CLIENT=multi` 时，默认角色是：

- `controller`: 读取当前实验计划，为生成模型分配研究方向
- `critic`: 审查 controller 计划，指出重复家族、弱锚点、地区阈值遗漏和模型分工问题
- `generator`: 发散生成候选
- `optimizer`: 围绕接近门槛的候选做优化型生成
- `validator`: 做 JSON、字段、格式和明显语法校验

服务端强制控制候选数量和回测设置。AI 可以提出 `settings`，但只会记录为 `proposed_settings`，不能覆盖实际用于 BRAIN 的 `region`、`universe`、`delay`、`neutralization`、`decay` 或 `truncation`。

每个候选都会记录模型来源、模型角色、模型名、controller allocation、validator metadata 和实验计划元数据。

## 约束内的机制迁移

系统保留这些约束：

- 大众字段如 `close`、`vwap`、`volume` 只能做辅助，不能当主信号
- 最近 approved/submitted 的核心字段要避开
- 当前季度已点亮塔尽量少重复探索
- 相关性和 data diversity 检查必须通过

当某个 scope 连续失败过多时，会进入困难模式：不直接放开约束，而是把历史高分但被阻断的候选转成“机制样本”。AI 只能学习机制，例如中周期平滑、行业内排序、scale normalization、revision/dispersion 逻辑，不能复制被禁止的字段或原表达式。

## 闭环模式

`BRAIN_CLIENT=local`：

- 生成候选
- 使用确定性的假回测
- 只用于测试服务代码

`BRAIN_CLIENT=http` 且 `AUTO_SUBMIT=false`：

- 生成候选
- 2-8 条表达式走 BRAIN multisimulation
- 单条表达式走普通 simulation
- 轮询 simulation URL，获取 alpha id 或平台错误
- 读取 `/alphas/{id}` 和 `/alphas/{id}/check`
- 写入真实指标和检查
- 只 approve，不 submit

`BRAIN_CLIENT=http` 且 `AUTO_SUBMIT=true`：

- 使用同一套真实回测和 guard
- 只有 guard 完全通过才提交
- 提交后确认 alpha 进入 `stage=OS` 且有 `dateSubmitted`
- 仍然遵守每轮最终提交上限

## Scope 预设

支持：

- `--preset us-d0` / `--preset us-d1`
- `--preset chn-d0` / `--preset chn-d1`
- `--preset eur-d0` / `--preset eur-d1`
- `--preset glb`
- `--preset ind`
- `--preset asi`
- `--preset hkg`
- `--preset kor`

也可以直接传：

```bash
--region USA --universe TOP3000 --delay 0 --neutralization SUBINDUSTRY --decay 6 --truncation 0.03
```

查看所有预设：

```bash
PYTHONPATH=src python3 -m alpha.cli presets
```

## 中转 API 地址

OpenAI-compatible 中转地址配置：

```bash
AI_CLIENT=openai
AI_API_KEY=your_relay_key
AI_BASE_URL=https://your-relay-host/v1
```

如果服务商给的是完整 endpoint，也支持：

```bash
AI_BASE_URL=https://your-relay-host/v1/chat/completions
```

不要只写 `/v1/chat/completions`，必须包含协议和域名。
