# PR #20 第四轮：修复方案

**评审来源**：Uzbekunknown，2026-10-24，在 Arc 140V 实测 `22f9667`
**阻塞项**：2 个（MinGW 配置失败、Linux 编译失败）
**非阻塞**：3 个（ImGui 无按键说明、网络两处选择、nr_enabled 写反）

**五项我全部独立核实过**，其中两处比作者的描述更严重、一处需要修正他的判断。下文每条都写清了实测证据。

---

## 阻塞一：MinGW 的 CMake 无法生成

### 作者的报错

```
CMake Error at CMakeLists.txt:301 (add_test):
  $<TARGET_FILE:test_device_teardown>   No target "test_device_teardown"
```

### 核实结果：**确认成立，且是同一处错误的第三次**

`CMakeLists.txt:332` 的 `if(WIN32)` 循环仍注册五个 fixture，但其中 `test_device_teardown` 和
`test_gpu_only` 的 `add_executable` 已随本轮移进 `if(NR_BUILD_LAYER)`（L246-249），而 `NR_BUILD_LAYER`
在 Windows 默认 OFF。

这正是我给 `test_transfer_resources` 修过的错误——当时只修了那一个，漏了这两个。

### 修复

按作者建议：**把这两个 fixture 的构建移回 `if(WIN32)`，并补上菜单源**。这样配置期和覆盖度
一起恢复。`test_transfer_resources` 的注册保持现在的条件化（它必须，因为它 `#include
"nr_layer.c"`）。

**同时**：这三个 fixture 现在在 Windows 上哪里都不构建（作者实测 CTest 从 65 降到 62）。修完后
应恢复 65。

---

## 阻塞二：Linux 因本 PR 而编译失败

### 作者的说法

> On `newnr`, `nr_layer.c` has none of them: `windows.h` is under `#ifdef _WIN32`, with a
> pthread arm beside it. The PR adds 16 lines of Win32, calls and `HWND`s, with no `#if`
> around them, at lines ~2477-2998.

### 核实结果：**确认成立，但比 16 行严重得多**

我实测了每个 Win32 token 的条件编译深度（脚本 `C:\temp\check_win32_depth.py`）：

```
L 2477 depth=0  CODE    GetAsyncKeyState      ← 行 2477
L 2498 depth=0  CODE    HWND
L 2515 depth=0  CODE    HWND
L 2521 depth=0  CODE    HWND
L 2533 depth=0  CODE    GetClientRect
L 2553 depth=0  CODE    GetForegroundWindow
L 2554 depth=0  CODE    GetAncestor
L 2562 depth=0  CODE    GetCursorPos
L 2581 depth=0  CODE    EnumWindows
L 2646 depth=0  CODE    GetCursorPos
L 2651 depth=0  CODE    ClientToScreen
L 2667 depth=0  CODE    GetWindowThreadProcessId
...

total lines mentioning a Win32 token: 42
of which CODE at depth 0 (unguarded): 27
```

**27 行代码在 `#if` 深度 0**，包括 5 个新增函数：`window_is_foreground`、`window_is_ours`、
`nr_window_enum_proc`、`window_for_thread_for_ui`，以及 6 处 `GetAsyncKeyState` 按键读取。

注意：`windows.h` 本身**仍在** `#ifdef _WIN32` 内（L116-117），所以 Linux 上不是“缺 include”
报错，而是**这些函数体内的类型和 API 未定义**。

### 修复

按作者建议：**给这五个函数和六处按键读加 `#ifdef _WIN32` 门控，非 Windows 提供 no-op arm**，
形状与 `nr_focus.h` 现有的 `static inline` 一致。

关键是**非 Windows 下语义不能错**：

| 函数 | Linux 下应为 |
|---|---|
| `window_is_foreground` | 返回 1（Linux 上图层就是唯一的 UI） |
| `window_is_ours` | 返回 1 |
| `window_for_thread_for_ui` | 返回 NULL → 其调用方必须能处理 |
| `nr_window_enum_proc` | 不需要 |
| 按键读取 | `nrex_key(..., 0)`（无按键） |

**风险点**：`window_for_thread_for_ui` 返回 NULL 时，L2622 附近的代码有 `if (!hwnd)
UI_MOUSE_STOP(...)` 分支——需要确认它不会在 NULL 时崩。

### 风险已评估：可控，两道门都会在 NULL 时安全返回

实测确认了两个安全点，所以门控不会把 Linux 变成“面板崩”：

1. **`menu_ui_mouse` 第 0 道门**（L2605）：`if (!data->menu_ui) UI_MOUSE_STOP(0, ...)`。
   非 Windows 下 `nrex_create` 编译为空实现，`data->menu_ui` 恒为 NULL，所以它**在第一行就
   返回**，根本走不到 `window_for_thread_for_ui`。
2. **`menu_ui_mouse` 第 2 道门**（L2623）：`if (!hwnd) UI_MOUSE_STOP(2, "no window for this
   process")`。即使将来 `menu_ui` 存在，NULL 窗口也会安全返回并打一条日志。

需要门控的完整清单（按 L2477-2998 实测）：

| 行 | 内容 | 非 Windows 处理 |
|---|---|---|
| 2498 | `window_is_foreground(HWND)` | 返回 1 |
| 2515 | `window_is_ours(HWND)` | 返回 1 |
| 2521 | `nr_window_enum_proc` | 整函数删除（仅 Windows 用） |
| 2550 | `window_for_thread_for_ui` | 返回 NULL |
| 2597-2670 | `menu_ui_mouse` 全体 | 保留：`!data->menu_ui` 会拦住它 |
| 2477 | `menu_ui_keys` 内的 `GetAsyncKeyState` | `nrex_key(..., 0)` |
| 2804/2921/2945/2984/2993/2998 | 6 处热键读 | 恒 0（或把整块包进 `#ifdef`） |

`nr_focus.h` 已提供 `nr_window_is_foreground` 的 `static inline` no-op 模式，是现成的范本。

---

## 非阻塞三：那一句按键说明，ImGui 面板看不到

### 作者的说法

> The line ending "the game also sees these keys" is drawn only by the quad menu.
> `nr_menu_ui.cpp` draws no hint at all, so the ImGui panel shows no key list.

### 核实结果：**核心成立，但描述需修正**

`nr_menu_ui.cpp` **确实画 hint** —— `nrex_hint_tooltip(r.hint)` 在 L959/972/983，画的是
**每一行的 tooltip**。

但那句“游戏也会收到这些按键”是 **`NR_STR_HINT`**，它是**面板级的提示文本**，只喂给手绘菜单：

```c
// nr_menu_bridge.c:1398
nr_menu_set_hint(b->menu, nr_menu_text(b->language, NR_STR_HINT));
```

而 `b->menu` 是**手绘菜单**（`nr_layer.c:2252` 传入的 `data->menu`）。ImGui 面板
（`data->menu_ui`，来自 `nrex_create`）**是另一个对象，它只接收 `r.hint`，不接收面板级
hint**。

所以：**ImGui 面板完全不显示面板级 hint，那三行按键列表玩家看不到。** 作者的结论对，机制与我
说法的细节有出入。

### 修复

两种选择，我倾向 A：

**A. 给 ImGui 面板加一个面板级 hint 行**（推荐）
在 `nrex_begin_panel` 或 footer 处增加一个可选的 hint 文本参数，`nr_layer.c` 把
`NR_STR_HINT` 传进去。改动面：`nr_menu_ui.h` 签名 + `nr_layer.c` 调用点 + `nr_menu_ui.cpp` 绘制。

**B. 把那句话塞进每行 hint**
更省事，但会在每一行 tooltip 里重复，不合适。

---

## 非阻塞四：网络在两个地方被选择

### 作者观察到的现象

> after Save with B chosen in MK11, `nr_settings.json` held `"weights": "cc_small_fp8"`,
> while `nr-env.txt` and the profile kept `NR_PRESET=A`. The next game, DoA5, loaded B
> (`weights: cc_small_fp8`, 31 blocks, history ceiling 0.767) and logged it as
> `network preset A`.

### 核实结果：**确认成立，且日志误导是设计使然**

- `nr_menu_bridge.c` **完全不写 `NR_PRESET`**（全仓仅 L25 注释提到）
- `NR_PRESET` 只由**窗口**写（`windows_wizard_core.py:405`）
- daemon 的优先级：**settings 的 `weights` 优先，`NR_PRESET` 仅兜底**（`nr_daemon.py:1522/1594`
  注释明说）

所以：DoA5 加载 B 是**正确**的（settings 优先），但日志写 “network preset A” 是**按 `NR_PRESET`
报的**，与实际加载不符。

### 修复

这是**你已决定最后阶段处理**的问题（`setup-network-box` 分支的 `weights` 旋钮 vs 作者的网络框）。
**本轮只修日志误导**：让 daemon 日志报**实际加载的**网络，而不是 `NR_PRESET` 的值。

完整合并留到最后阶段。

---

## 非阻塞五：Save 写了 `"nr_enabled": 0`，而 NR 是开的

### 核实结果：**确认成立**

`nr_enabled_saved` 只在两处更新：

- `nr_menu_bridge.c:1286` —— `nr_menu_bridge_set_nr_saved`，**由 Home 键调用**
- L1057 —— 从文件读回

**窗口的「启用 NR」按钮不调它**，所以面板侧的 `nr_enabled_saved` 保持旧值，Save 时写出的就是
反的。

### 修复

**窗口的 Enable/Disable 也应调用 `nr_menu_bridge_set_nr_saved`**，或更简单：面板每次刷新时
**从 trigger 文件的真实状态同步** `nr_enabled_saved`，而不是依赖按键路径。

后者更稳，因为 trigger 才是唯一真相（`nr_layer.c` 里 `nr_menu_set_nr_enabled` 已经是这么做的）。

---

## 执行顺序

```
阻塞二（Linux 门控）  ← 最大工作量，先做，因为它会改 nr_layer.c 结构
    ↓
阻塞一（MinGW 配置）  ← 独立，可与上面并行
    ↓
非阻塞三（ImGui hint 行）  ← 依赖 nr_menu_ui 的签名改动
非阻塞五（nr_enabled 同步）  ← 独立
非阻塞四（日志误导）        ← 独立
```

**依赖图**：非阻塞三需要在阻塞二之后（都改 `nr_layer.c` 的菜单区域），其余可并行。

---

## 验证门槛

| 阶段 | 必须通过 |
|---|---|
| 阻塞二 | MSVC 构建 + 9/9 fixtures + 257 passed；**Linux 需作者或 CI 验证** |
| 阻塞一 | 作者实测 MinGW 配置通过、CTest 恢复到 65 |
| 非阻塞三 | ImGui 面板在游戏内显示那三行按键（作者实测） |
| 非阻塞五 | 窗口启用 NR 后 Save，文件写 `nr_enabled: 1` |

**本机局限**：无 gcc/clang/make/32 位工具链，Linux 与 MinGW 仍需作者代验。这一点会在提交
信息和汇报里如实说明。
