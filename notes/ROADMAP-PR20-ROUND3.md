# PR #20 第三轮评审：路线书

**状态**：计划中，未开始实施
**基线**：`menu-on-e4m3` @ `da53550`
**评审来源**：Uzbekunknown，2026-10-22，13 条
**本文用途**：确定每条的顺序、依赖和验证点。不是给读者的文档，是给实施者的施工图。

---

## 0. 一句话

13 条里 **6 条是 Windows 上可直接验证的缺陷**，**2 条是架构层面的错误**
（构建系统与 ABI），**1 条是设计冲突**（pacing 归属），**4 条是小项**。
排序的第一原则是：**先让所有构建通，否则其它修复无法验证**。

---

## 1. 阶段划分

### 阶段 0：让构建重新通（阻塞其它一切）

**为什么最先做**：作者报的三个 MinGW 失败、Linux 链接失败、
以及 5 个 `#include "nr_layer.c"` 型测试，都来自同一个根因。
在这个前提下，其它修复无法用 CTest 验证。

| 条目 | 任务 | 依赖 |
|---|---|---|
| 5a | `nr_menu_ui.cpp` 加平台门控 | 无 |
| 5b | CMake 补菜单源到 layer 与测试目标 | 5a |
| 5c | Makefile 同样补 | 5a |
| 4 | 写 `test_menu_keys.py` 或撤掉注册 | 5b（它要能被 CTest 跑到）|
| 4b | 把 `test_focus_gate.c` 接进 Windows 构建 | 无 |

**关键发现（侦察已确认）**：
- 5 个测试用 `#include "nr_layer.c"`（`test_device_teardown.c:25`、
  `test_exchange.c:8`、`test_gpu_only.c:22`、`test_settled.c:6`、
  `test_transfer_resources.c:3`），因此继承了 layer 对菜单的全部依赖
- `CMakeLists.txt:153/171` 的 `nr_layer`/`nr_layer32` 是单文件目标
- `Makefile:87-88/137-138` 同样单文件；`MENU_SOURCES`（L91-92）只被
  三个菜单测试用，**layer 目标从不用**
- `test_gpu_present`（CMake L219）已用 `add_dependencies(... nr_layer ...)`，
  **链接库是这个项目已有的模式**，5 个 `#include` 型测试只是没跟进

**4b：`test_focus_gate.c` 从未在 Windows 上跑过**
- 文件存在（85 行），接在 `Makefile:130-131`，由 `make test` 跑
- **`CMakeLists.txt` 里没有目标，`tools/build_win.bat` 里没有编译行**
- 它是第 2 条修复唯一依赖的规则（焦点判定）的测试，
  却在 Windows 构建里完全不运行
- 这是"检查存在但没被跑"的又一类问题，与第 4 条同源

**5a 的具体内容**：`nr_menu_ui.cpp` 的 Windows 专有无门控段：
- L647-651 `LARGE_INTEGER` + `QueryPerformanceFrequency/Counter`
- L1316-1339 `menu_ui_keys`：`GetAsyncKeyState`、`BYTE state[256]`、
  `SHORT`、`WCHAR`、`ToUnicode`
- 处理方式：`#ifdef _WIN32` 完整实现；`#else` 提供
  `clock_gettime(CLOCK_MONOTONIC)` 计时与 no-op 按键喂入，
  使 `nrex_create` 在 Linux 返回 `nullptr`（菜单不存在），
  与 `nr_menu.h:28-31` 承诺的"拒绝即没有菜单，不是坏游戏"一致
**验证点**：
- Windows MSVC 双架构构建通过，7 个菜单 fixture 全绿
- MinGW 构建通过，CTest 65/65（或明确记录哪些跳过）
- Linux `make test` 通过，`libnr_layer.so` 无未定义符号（`nm -u`）
- CTest 无 "can't open file" 条目

---

### 阶段 1：32 位正确性（阻塞合并）

**为什么第二**：作者明确说"DoA5 32-bit crashes ... a 32-bit game needs one before merge"。
这是**唯一一条被明确标为合并阻塞**且涉及崩溃的。

| 条目 | 任务 |
|---|---|
| 1 | x86 ABI：`VKAPI_PTR`、非可分派句柄宽度 |

**关键发现（侦察已确认）**：
- `nr_menu_ui.cpp:122` `typedef PFN_vkVoidFunction (*nrex_gpa_fn)(void *device, const char *name);`
  **无 `VKAPI_PTR`** → x86 上每次调用不平衡栈 8 字节
- 作者的后续怀疑：x86 上 `VkImage`/`VkImageView` 是 `uint64_t`，
  而 `nrex_image`/`nrex_image_view` 返回 `void *` → 截断
- `nr_menu_ui.cpp:260-270` 用了 `VK_NULL_HANDLE` 而非 `nullptr`，
  注释说明作者已意识到 32 位句柄宽度问题（"pointers only when... 32-bit games"）

**实施顺序**：
1. 加 `VKAPI_PTR`（最小改动，先验证能否解决）
2. 若仍崩溃，把 `nrex_image`/`nrex_image_view` 的返回类型改为
   `VK_DEFINE_NON_DISPATCHABLE_HANDLE` 等价物（`uint64_t`）
3. 检查 `ctx[0]`/`ctx[1]` 把函数指针装进 `void *` 的位置（L483-484）

**验证点（必须真机）**：
- DoA5（DX9，32 位）+ `NR_MENU` 开：连续启动 3 次不崩溃
- 同安装 `NR_MENU=0` 作为对照：40ms/帧 @1024x768
- 32 位下菜单能开、能走行、能存档

---

### 阶段 2：输入正确性

**为什么第三**：第 2、3 条都是"功能声称完成但实际半成品"，
且第 3 条有明确的错误返回值可验证。

| 条目 | 任务 |
|---|---|
| 3 | `GetRegisteredRawInputDevices` 的 `cbSize` |
| 2 | `nr_menu.c` 的焦点门控 |
| 7 | 面板按键透传（若无法修复，加提示） |

**第 3 条（最小、最确定）**：
- `nr_raw_mouse.c:66` 和 `:70` 传 `sizeof(RAWINPUTHEADER)`，
  应为 `sizeof(RAWINPUTDEVICE)`
- x64：24 vs 16 → 返回 `(UINT)-1` / `ERROR_INVALID_PARAMETER`(87)
- **关键陷阱**：`(UINT)-1` 是**真值**，所以现有的
  `if (GetRegisteredRawInputDevices(...))` 把**失败当成功**。
  判断必须写 `== (UINT)-1` 或 `== 0xFFFFFFFF`，不能写 `!ret`
- 同一文件里已有正确写法可抄：`nr_raw_mouse.c:146`
  `if (count == 0 || count == (UINT)-1) return 0;`
- 还需处理 `ERROR_INSUFFICIENT_BUFFER`（用更大的 count 重试），
  且失败时不要用未初始化的 `devices` 缓冲跑循环
- `nr_raw_mouse_stop:128` 的恢复返回值也被忽略——注释已承认恢复可能失败，
  但失败是静默的；加一行 trace 成本很低
- **测试**：新建 `src/layer/test_raw_mouse.c`——自建隐藏窗口，
  先注册 page1/usage2（模拟游戏）**并注册 page1/usage6（键盘对照）**，
  跑 start/stop 后重新读取，断言：
  - usage2 条目存在，`hwndTarget` 与原 flags 一致（恢复成功）
  - usage6 条目仍在（键盘未被动过）
  - `sizeof(RAWINPUTDEVICE) != sizeof(RAWINPUTHEADER)`（防未来改回）
- **注意**：这个 fixture 能验证代码逻辑，**不能**验证游戏进程内的真实恢复
  （注册表是每进程的），后者仍需 MK11/DoA5 手测

**第 2 条（有约束）**：
- 根因：`nr_menu.c` **完全不 include `nr_focus.h`**（include 只有
  nr_menu.h/strings/font + 标准库 + windows + vulkan）
- 现状（逐行核对 `nr_menu_update`，L891-943）：

  | 行 | 键 | 现有门控 |
  |---|---|---|
  | 906 | INS | `m->read_keys` |
  | **907** | **Home** | **完全没有** |
  | 922-923 | S / R | **只有 `m->open`** |
  | 928-939 | 方向/Enter/L/Esc | `m->open && m->read_keys` |

- **关键：ImGui 构建下 `read_keys == 0`**（`nr_layer.c:2223`
  调 `nr_menu_set_key_reading(data->menu, 0)`）。这杀掉了 928-939，
  但 **906/907/921-924 全部存活且无门控**。
  所以 `nr_layer.c:2934` 那个已门控的 Home **保护不了它** ——
  `nr_menu.c:907` 是同一个 `toggle_pressed` 的**第二个无门控所有者**。
  **只改图层层不够，`nr_menu.c` 必须一起改。**
- **硬约束（项目所有者明确要求）**：
  **Home 必须在面板打开和关闭时都工作。** 修复只加焦点门控，
  **不能**改成"仅面板打开时生效"
- 现成工具：`nr_focus.c:15-30` 的 `nr_window_is_foreground(void *hwnd)`，
  `nr_focus.h:45-46` 在非 Windows 有 `static inline` no-op（返回 1）
- **实施方案（不改 `nr_menu.c` 的焦点判断，改用既有开关）**：
  1. `nr_layer.c` 在 `menu_read_keys` 里，`foreground` 已算好（L2850）。
     在调用 `nr_menu_update`（L2964）之前：
     `if (!foreground) nr_menu_set_key_reading(data->menu, 0);`
     `else if (!data->menu_ui) nr_menu_set_key_reading(data->menu, 1);`
     —— 复用**已经存在的机制**（L2223 就是这么用的），不新增 API
  2. `nr_menu.c:907` 加 `m->read_keys &&`
  3. `nr_menu.c:921` 改成 `if (m->open && m->read_keys)`
  4. 为什么这满足"面板开关都工作"：`read_keys` 在**手绘构建里恒为 1**
     （面板开关都行），在 **ImGui 构建里恒为 0**
     （Home 由已门控的 `nr_layer.c:2934` 读）。两种构建都不变成
     "仅面板打开时生效"
- **必须做的第四步（易漏，漏了会变成更严重的 bug）**：
  `key_pressed()` 即使结果被丢弃**也会锁存** `m->key_down[]`（L279-280）。
  所以门控期间按住 Home，`was=1` 被记下，回到游戏后第一次按
  **会被吞掉**。需要新增 `nr_menu_release_keys(m)`（一个 `memset`），
  在 `!foreground` 时调用。项目已经吃过这个亏——
  `nr_layer.c:2880-2883`、`2904-2906` 的注释就是在说这件事
- **顺带要修的注释**：`nr_layer.c:2910` 说"All three are inside
  `nr_menu_update`'s `m->open && m->read_keys` block"——**已经是假的**
  （S/R 被移出来了）；`nr_menu.c:909-920` 的论证前提也已不成立
- **死代码（无害但不该留着）**：`NR_KEY_SHIFT`(0x10)、`NR_KEY_C`(0x43)
  定义了从不读取；`key_down()`（L288-297）定义从不调用
- **验证**：`test_menu_keys.cpp` 已有的"读源码做结构断言"机制
  最适合承载（它已经在断言 S/R 在 guard 之上、Home 是 0x24 等）。
  加：`nr_menu.c` 里 Home 的读取带上 `read_keys` 门；
  `nr_layer.c` 里有 `nr_menu_set_key_reading` 且参数不是字面量。
  **注意**：`test_menu_keys.py` 不存在（第 4 条），
  而 `test_menu_keys.cpp` 才是那个承载者

**第 7 条（诚实处理）**：
- 呈现层的按键无法从游戏手里拿走（`GetAsyncKeyState` 是系统级，
  没有"消费"操作；仓库里没有 `SetWindowsHookEx`/`RegisterHotKey`/
  `BlockInput`/`SendInput`，已 grep 确认）
- 三个候选手段都不可接受：
  - `RegisterHotKey`：系统级独占，破坏游戏自身绑定
  - `WH_KEYBOARD_LL`：唯一能真正抑制的，但需消息泵，
    而图层跑在游戏渲染线程上；且全局钩子杀毒软件会拦
  - 子类化游戏 `WndProc`（OptiScaler 的做法）：本层明确不做
- 采用作者的退路：**面板里加一行说明**。位置应加在已有的提示串里
  （`nr_menu_strings.py` 的 `hint`，生成表，不要手改 `.c`），
  现有内容已列出面板自己的键：
  `"Up/Down choose  Left/Right change  L language  Home NR on/off  S save  R reset  INS close"`
- 诚实的表述：**面板打开时游戏仍会收到导航键；按 INS 关掉面板再玩**
- 不做假承诺

---

### 阶段 3：pacing 归属（设计冲突，需一次性厘清）

**为什么单独一阶段**：第 6 条不是 bug 而是**上一轮设计决策的后果**，
修它需要重看两个写入者的关系。

| 条目 | 任务 |
|---|---|
| 6 | `set_pacing` 恢复写 `nr-env.txt`，与 profile 同步 |

**背景（必须理解）**：
- 上一轮为消除"两个写入者互相覆盖"，把 `set_pacing` 改成只写 profile
- 结果：窗口报 `applied: true`（"下次启动生效"），
  但游戏实际要重新 Install NR 才变 —— 作者的发现
- 面板侧已有 `write_profile_pacing`（`nr_menu_bridge.c`），
  实测在 MK11 生效：面板写文件 + 写 profile

**`applied` 到底是什么（关键澄清）**：
`windows_wizard_core.py:1643`
```python
applied = owns and (folder / OWNER_NAME).is_file() and (folder / ENV_FILE).is_file()
```
这是**"安装存在且归本版本所有"的事实**，不是"写了文件"。
而窗口字符串（`windows-wizard.ps1:501`）说
"Frame pacing saved; it takes effect when the game starts next." ——
**它把一个存在性判断当成了写入承诺**。这是缺陷的准确形状。

对照 `:502` 的 `Done.pacing-later` =
"Frame pacing saved; Install NR gives it to the game." ——
这句是诚实的，但它只在 `applied` 为假时显示。**两句的触发条件反了**：
该说"要重新安装"的时候说了"下次生效"。

**正确设计（作者建议，采纳）**：
- 面板：写 `nr-env.txt` + 写 profile（现状，保留）
- 窗口：**也写两边**——用与原文件同形的就地改写（只替换 `PACING`
  六行，其余逐字节保留），然后原子替换（`.new` + `os.replace`）
- 不能整份重新生成：那需要 `NR_GAME_EXE`、`NR_LAUNCH_STATE`、
  条件性的 `NR_REAL_VULKAN`，而 `nr_menu_bridge.c:466-469` 明确警告
  丢失这些行会破坏安装
- 写完 `applied` 保持当前含义，但它**现在真的**意味着"下次启动会生效"

**残留风险（要写进注释，不是忽略）**：
两边都用"读-改-写"且无锁。真正的并发写（面板在游戏里，窗口在桌面）
可能丢更新，但原子替换保证最坏是"后写者赢"，不会出现半个文件。
面板每次 `load_settings`/refresh 都重读 `nr-env.txt`（`nr_menu_bridge.c:719`、
`:1256`），所以窗口改完、面板下次打开会看到窗口的值——**一致，不是静默错误**。

**需要改的测试（四处，全部翻转）**：
| 位置 | 现在断言 | 改成 |
|---|---|---|
| `test_windows_wizard.py:384-385` | 文件不变 | 文件被写，且非 pacing 行逐字节不变 |
| `:429` | `path.read_text() == before` | GPU 行变 1 |
| `:435` | 同上 | GPU 行变 0 |
| `:482` | `NR_PRESET == "A"` | 成功保存后为 `"B"` |

**新增断言（模仿 `test_menu_pacing.cpp:508` "两边一致"）**：
读回 `nr-env.txt` 的 pacing 行，逐个与 `layer_environment(saved_profile)`
比对。

**验证点**：
- 面板改 pacing → 窗口打开 → 显示的是面板的值
- 窗口改 pacing → 下次启动生效（无需重新 Install）
- 面板改完 → 窗口再存一次 → 不覆盖（关键回归）

---

### 阶段 4：小项批量清理

| 条目 | 任务 | 复杂度 |
|---|---|---|
| 8 | 俄语截断 + 复数形式 | 小 |
| 9 | `nr_layer.c:88` 换行 | 极小 |
| 10 | `trace_launch` 改 opt-in + 修 append | 小 |
| 11a | `check_imgui_pristine.py` 去掉硬编码路径 | 极小 |
| 11b | 面板 Network 行显示可读名 | 小 |
| 12 | 去掉 "the owner" 归因 | 极小 |

**第 8 条细节**：
- `nr_layer.c:2333` `static char every_names_store[4][24]`
- "каждые 2 кадров"：**15 个字符，但 27 个 UTF-8 字节** → `snprintf` 按字节截断，
  得 `каждые 2 кадр`（实测，与作者报告一致）

- **实测四种语言 × 四个档位**（`nr_menu_strings.py` 的真实字符串）：

  | 语言 | `every_frame` | `every_nth` 建出的最长 | 24 字节够吗 |
  |---|---|---|---|
  | en | 11 B | `every 2 frames` = 14 B | ✅ |
  | zh | 6 B | `每 2 帧` = 9 B | ✅ |
  | **ru** | 21 B | **`каждые 2 кадров` = 27 B** | ❌ **三个档位全溢出** |
  | es | 14 B | `cada 2 fotogramas` = 17 B | ✅ |

  **只有俄语溢出，且 c=1/2/3 全部溢出**（不是某一档）。

- **注意**：按字符数算会得出"放得下"的错误结论（15 字符 × 1 字节 = 15）。
  俄语西里尔字母每个 2 字节，中文每字 3 字节——这类判断一律按字节。
  这个坑真实发生了：并行调查的第一版报告就用字符数算出"放得下"，
  后来自己实测才更正
- 俄语复数规则：1 кадр / 2-4 кадра / 5+ кадров
- 当前机制：`NR_STR_EVERY_NTH` 用 `{}` 占位，`nr_layer.c` 里替换
- **两处都要改，是两个独立的 bug**：
  - 缓冲区 `[4][24]` → `[4][48]`
  - 复数形式 → 加两个键（`every_nth_few` / `every_nth_many`），
    与现有 `every_frame`（1）凑成三档，由 C 按数字选
- **修正后的俄语实测字节数**（`[]` 内是可用的 23 字节上限）：

  | 键 | 模板 | 填入数字后 |
  |---|---|---|
  | `every_frame`（1） | `каждый кадр` 21 B | 21 B ✅ |
  | `every_nth_few`（2-4） | `каждые {} кадра` 26 B | **25 B** ❌ |
  | `every_nth_many`（5+） | `каждые {} кадров` 28 B | 27 B ❌ |
  | `every_nth`（现状） | `каждые {} кадров` 28 B | 27 B ❌ |

- **关键**：改完复数形式后 2-4 档仍是 **25 字节**，依然超过 23。
  所以缓冲区必须一起改，**两个改动不能互相替代**
  （这一点容易误判成"改了复数就好了"）
- **不要顺带改 `NR_KNOB_CHOICE_NAME_SIZE`**（`nr_menu_bridge.h:223` 的 32）：
  它装的是旋钮原始名（`ht`、`cc_small_fp8`、`standard`，≤12 字节），
  与 `nr_layer.c` 的 cadence 数组是两个独立的东西
- `snprintf` 截断是静默的：检查返回值 `>= sizeof`，超长时回退到
  `shape` 原文而不是显示半截词
- **教训记录**：这条的字节数在调查过程中被算错过两次（都把西里尔字母
  当 1 字节），两次都是实测才纠正。写这类字符串一律 `len(s.encode())`，
  不要数格子

**第 10 条细节**：
- `nr_vulkan_proxy.c` 的 `trace_launch` 从 `DllMain` 运行
- **是两处缺陷**：
  - 用了 `GENERIC_WRITE | FILE_APPEND_DATA`——`GENERIC_WRITE` 含
    `FILE_WRITE_DATA`，后者抵消了 append，每次写到偏移 0 → 只留最后一行
  - 无条件运行（每个加载代理的进程）
- 改：
  - `GENERIC_READ | FILE_APPEND_DATA`（`record_start_folder` 已经是正确写法，
    照抄它）
  - 加 `NR_PROXY_TRACE=1` 开关（参照 `NR_MENU_DUMP` 的 opt-in 风格，
    而不是 `NR_MENU=0` 的 opt-out——后者是每帧成本，前者是每次启动）
  - stderr 那半也同样受开关控制
- **铁证**：同一个文件里两种写法并存——
  - `nr_vulkan_proxy.c:198` `record_start_folder`：
    `GENERIC_READ | FILE_APPEND_DATA`，注释 "Without FILE_WRITE_DATA every
    write lands at the end." ← 正确
  - `nr_vulkan_proxy.c:232` `trace_launch`：
    `GENERIC_WRITE | FILE_APPEND_DATA`，注释 "Appended, so a run of launches
    can be compared." ← 错误且注释在撒谎
  `GENERIC_WRITE` 展开含 `FILE_WRITE_DATA`，后者抵消 append 语义。

**第 11b 条细节**：
- 面板的 Network 行显示**原始名** `ht` / `cc_small_fp8`；
  窗口显示 `A: HT, the model this project reproduces` /
  `B: CC small FP8, the faster model`
- **根源**：`nr_menu_bridge.c:188` 的旋钮定义里 choices 就是原始名，
  `choice_name()`（L274-298）按逗号切分后原样返回；
  `format_row()`（L308-317）把它写进 `shown[i]` → 成为行的 value
- 面板这条路的**标签和提示是翻译的**（`nr_menu_strings.py` 的
  `weights` 标签与 hint），**只有 value 不是**
- **字符串已经有了，四种语言齐全**，就在安装窗口的表里：

  | value | en | ru | es | zh |
  |---|---|---|---|---|
  | `ht` | A: HT, the model this project reproduces | A: HT, модель, которую воспроизводит проект | A: HT, el modelo que reproduce el proyecto | A：HT（本项目复现的模型） |
  | `cc_small_fp8` | B: CC small FP8, the faster model | B: CC small FP8, быстрая модель | B: CC small FP8, el modelo rápido | B：CC small FP8（更快的模型） |

  （`windows-wizard.ps1:530/633/736/839`，四条 `Choice.weights.*`）

- **不存在的部分**：生成的菜单表 `nr_menu_strings.py` **一个 `Choice.*`
  都没有**（实测 0 个）。生成器 `tools/gen_menu_strings.py` 只映射
  旋钮标签、提示和自己的 `OWN` 键，**不处理 choice 值**
- **修法**：在生成器的 `EXTRA` 里加两条
  （`choice_weights_ht` → `("Choice.weights.ht", ...)` 与
  `choice_weights_cc_small_fp8`），生成器会从安装窗口的四语言表自动
  带出 ru/es/zh，**不需要新翻译**，也保证两个 UI 不会漂移
- **关键不变式**：设置文件和 daemon 必须继续收到**原始名**
  （`nr_daemon.py:1606`、`nr_weight_models.py:71` 按那个拼写解析）。
  只有**显示**变。写入路径读的是 `b->choice[i]`（`nr_menu_bridge.c:889`、
  `:1021`、`:1050`），那里必须保持原始名。所以 `choice_label` 必须是
  **仅用于显示**的，与 `label_for` 对行名的角色一致
- **宽度**：`shown[NR_KNOBS_COUNT][96]`（`nr_menu_bridge.c:202`）够放
  （最长 zh 约 40 字节）；但要实际看一眼面板行宽，短了就只保留
  `A:`/`B:` 前缀 + 模型名，完整句子交给 hint（hint 已经写了）
- **测试**：给面板侧加一条"每个 choice 旋钮的每个值在四种语言里都有
  非空标签"——这正是 `test_windows_wizard.py:892-918` 对窗口做的检查，
  **面板侧没有对应检查，所以这个缺陷才漏出来**
- 参考落点：`test_menu_rows.py`（编译探针读 C 表）或
  `test_menu_persist.c:511-520`（已经遍历了所有语言 × 所有键查空）

---

### 阶段 5：交付基线修正

| 条目 | 任务 |
|---|---|
| 13 | PR base 改 `newnr`；`011dc98` 署名 |

**侦察发现（重要）**：
- PR 当前 base = `master` @ `2f90064`
- 要求改 base = `newnr`
- **本地 `menu-on-e4m3` 已与 `upstream/newnr`（`2d1d550`）完全相同
  （0 ahead / 0 behind）**，同时领先 `upstream/master` 79 个提交
- 所以这不是"重新 rebase 30 个提交"，而是**推送状态与 PR base 设置**的问题
- 需要：确认远端分支实际指向 → 改 PR base → 确认 diff 只剩本轮改动
- `011dc98` 作者 `Paimon <noreply@github.com>`；作者同意 squash on merge

---

## 2. 依赖图

```
阶段0 (构建)
  ├── 5a nr_menu_ui.cpp 门控
  ├── 5b CMake  ← 5a
  ├── 5c Makefile ← 5a
  ├── 4  test_menu_keys.py ← 5b
  └── 4b test_focus_gate 接进 Windows 构建

阶段1 (x86)  ← 阶段0（否则 32 位构建也断）
  └── 1 VKAPI_PTR → 句柄宽度

阶段2 (输入)  ← 阶段0
  ├── 3 raw mouse cbSize
  ├── 2 焦点门控 ← 依赖 4b（focus 规则在 Windows 上被测过）
  └── 7 提示行

阶段3 (pacing)  独立，但改测试需 阶段0 的测试能跑
  └── 6

阶段4 (小项)  独立
  ├── 8 俄语（缓冲 + 复数，两个独立 bug）
  ├── 9 换行
  ├── 10 trace_launch（append + opt-in，两个缺陷）
  ├── 11a 路径
  ├── 11b 面板标签
  └── 12 归因（与 6 同段，一起改）

阶段5 (交付)  最后
  └── 13 base + 署名
```

**关键路径**：阶段0 → 阶段1 → 阶段2 → 阶段3 → 阶段5。
阶段4 可任意穿插。

**注意交叉依赖**：第 6 条与第 12 条改的是同一段注释
（`nr_menu_bridge.c:530-533`），必须一起改——把 `set_pacing`
改成写两边之后，"`set_pacing` no longer needs to write the file at all"
这句就变成假的了。

---

## 3. 每阶段的验证门槛

| 阶段 | 必须通过的验证 |
|---|---|
| 0 | MSVC 双架构、MinGW、Linux `make test`、`nm -u` 无未定义、CTest 无缺文件；`test_focus_gate` 在 Windows 上真的跑起来 |
| 1 | DoA5 32 位连续 3 次不崩；`NR_MENU=0` 对照 40ms |
| 2 | 新 `test_raw_mouse.c` 恢复注册（目标窗口+flags）且键盘注册存活；焦点门控三态（前台开/前台关/后台）且**回到游戏后第一次按 Home 生效**（锁存未被吞） |
| 3 | 双向一致性：面板改→窗口读对，窗口改→下次生效，面板改→窗口存不覆盖 |
| 4 | 各自单元测试；俄语四档显示正确（含 2-4 档用 кадра）；面板 Network 行显示可读名且写盘仍是原始名 |
| 5 | PR diff 只含本轮改动；CI 绿 |

---

## 4. 明确不做

- **不**抽 dispatch 层解耦菜单（78 个调用点，风险远高于收益）
- **不**在阶段0 之前动任何其它文件（否则无法验证）
- **不**碰网络框（`.​.` 已隔离在 `setup-network-box` 分支，最后阶段处理）
- **不**承诺面板能阻止按键透传给游戏（物理上做不到）

---

## 5. 与作者沟通的节点

| 时点 | 内容 |
|---|---|
| 阶段0 完成 | 报告 MinGW/Linux/CTest 结果，说明 `nr_menu_ui.cpp` 门控的范围 |
| 阶段1 完成 | 报告 DoA5 32 位 3 次运行结果 |
| 阶段2 完成 | 报告 focus gate 的设计（强调 Home 面板开关都工作） |
| 阶段3 前 | **需确认** pacing 双向写入的测试断言改法 |
| 阶段5 前 | 确认 squash 与 base 切换顺序 |
| 全部完成 | 逐条回复 13 项，未修的明确说明原因 |

---

## 6. 风险登记

| 风险 | 影响 | 缓解 |
|---|---|---|
| `nr_menu_ui.cpp` 门控工作量超预期 | 阶段0 拖延 | 先只让 Windows 通，Linux 门控拆到独立提交 |
| x86 加 `VKAPI_PTR` 仍崩溃 | 阶段1 阻塞 | 作者的第二个怀疑（句柄宽度）是备选；需要有 32 位测试环境 |
| pacing 双向写入再引入覆盖 | 回归 | 必须先有"面板改→窗口存不覆盖"的测试 |
| 俄语复数需要改字符串表结构 | 触及生成器 | 用三个键（1/2-4/5+）而不是改格式串机制 |
| 无 Linux 环境 | 阶段0 的 Linux 验证做不了 | 明确告知作者，请其代跑；或只做 Windows 侧并说明 |
