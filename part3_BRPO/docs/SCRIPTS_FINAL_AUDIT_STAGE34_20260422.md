# SCRIPTS_FINAL_AUDIT_STAGE34_20260422

> 更新时间：2026-04-22 20:04 (Asia/Shanghai)
> 范围：scripts final audit 的 Stage 3（compat shim 单独处理）+ Stage 4（最终文档收尾）
> 结论：`run_pseudo_refinement.py` 已被收窄为外部 CLI wrapper；内部调用已全部切到 `scripts/compat/run_pseudo_refinement.py`；scripts 侧代码整理现在可以视为完成。

---

## 1. Stage 3：compat shim 收口

本轮不再把 `run_pseudo_refinement.py` 当成普通顶层 utility，而是把它明确拆成两层：

- 内部兼容入口：`scripts/compat/run_pseudo_refinement.py`
- 顶层外部 wrapper：`scripts/run_pseudo_refinement.py`

其中：
- `scripts/compat/run_pseudo_refinement.py` 负责真实加载 `scripts/archive_experiments/legacy_entry/run_pseudo_refinement.py`
- 顶层 `scripts/run_pseudo_refinement.py` 只保留极薄外壳，用于维持旧 CLI 路径稳定
- `scripts/run_pseudo_refinement_v2.py` 与 `scripts/diagnostics/diagnose_stageA_gradients.py` 已改为直接指向 `scripts/compat/run_pseudo_refinement.py`

这意味着：顶层 wrapper 不再承担内部调用职责，只是一个显式保留的 legacy facade。

---

## 2. Stage 4：文档收尾

本轮已同步更新：
- `docs/current/STATUS.md`
- `docs/current/DESIGN.md`
- `docs/current/CHANGELOG.md`
- `docs/agent/hermes.md`
- `scripts/README.md`
- 新增 `scripts/compat/README.md`
- 新增本记录 `docs/SCRIPTS_FINAL_AUDIT_STAGE34_20260422.md`

文档口径统一为：
- 顶层 `scripts/` = 8 个 live core + 1 个外部 CLI wrapper
- 内部 compatibility boundary = `scripts/compat/`
- diagnostics / legacy prepare / archive_experiments 各自独立收纳

---

## 3. 验证

### 3.1 `py_compile`
已覆盖并通过：
- `scripts/compat/run_pseudo_refinement.py`
- `scripts/run_pseudo_refinement.py`
- `scripts/run_pseudo_refinement_v2.py`
- `scripts/diagnostics/diagnose_stageA_gradients.py`

### 3.2 最小 real import / smoke
真实导入成功：
- `scripts/compat/run_pseudo_refinement.py`
- `scripts/run_pseudo_refinement.py`
- `scripts/run_pseudo_refinement_v2.py`
- `scripts/diagnostics/diagnose_stageA_gradients.py`

真实检查成功：
- `run_pseudo_refinement_v2.load_v1_module()` 能加载 compat 层并得到 legacy module
- `diagnose_stageA_gradients.load_v1_module()` 能加载 compat 层并得到 legacy module
- 顶层 `scripts/run_pseudo_refinement.py` 仍能作为外部 wrapper 导入并转发到 compat 层

### 3.3 路径审计
live 代码中不再有内部调用直接依赖顶层 `scripts/run_pseudo_refinement.py`；它只作为外部兼容 CLI 路径保留。

---

## 4. 收尾清单结论

本轮后 scripts 侧收尾清单已经全部完成：
1. Stage 1：diagnostics 下沉 —— 完成
2. Stage 2：legacy prepare 下沉 —— 完成
3. Stage 3：compat shim 单独处理 —— 完成
4. Stage 4：文档与收尾清单同步 —— 完成

当前顶层 `scripts/` 最终形态：
- live core：8 个
- 外部 CLI wrapper：1 个（`run_pseudo_refinement.py`）
- 不再有其它未分类的 non-live 脚本停留在顶层

---

## 5. 当前结论

- scripts 侧代码整理已完成
- pseudo_branch 第二轮整理已完成
- 如果后续还要继续做“工程整理”，优先级已不再是代码路径收纳，而是：
  1. backend-only integration 主线实现
  2. 视需要做 working tree / commit / archived-doc hygiene 的最后收尾
