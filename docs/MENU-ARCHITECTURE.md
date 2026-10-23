# 菜单层架构书

**范围**：`src/layer/` 下与游戏内设置菜单相关的部分
**读者**：要修改这一层的人，以及要判断某个改动是否安全的人
**本文与前作的关系**：`docs/ARCHITECTURE.md` 讲的是**网络本身**（图、权重、数值），
不涉及这一层。`docs/WINDOWS-PORT.md` 是按时间记录的移植史。
本文是**第一次把这个子系统当作一个架构来描述**。

---

## 1. 为什么这一层存在，为什么它长成现在这样

### 1.1 约束比设计先来

Vulkan 图层**寄生在别人的进程里**。它没有自己的窗口、没有消息循环、
没有 `WndProc` 可以子类化。这一条约束推出了下面几乎所有结构：

| 想做 | 正常人怎么做 | 图层只能怎么做 |
|---|---|---|
| 知道自己是否在前台 | `WM_SETFOCUS` / `WM_KILLFOCUS` | 比较 `GetForegroundWindow()` 与本进程窗口（`nr_focus.c`） |
| 读按键 | 窗口消息 | 每帧 `GetAsyncKeyState` 轮询（`nr_menu.c:273`） |
| 知道自己鼠标在哪 | `WM_MOUSEMOVE` | 系统光标位置 + 自行换算（`nr_menu_ui.cpp`） |
| 画 UI | 创建窗口 | 用 compute pass 往游戏的 swapchain 图像上画（`nr_menu.h:1-9`） |
| 用 ImGui | 给 ImGui 一个 `GLFWwindow` | 自己实现 ImGui 的 Vulkan 后端接线（无平台层） |

**参考项目 OptiScaler 用子类化解决了前两项**（`menu/input/input_system_messages.cpp`）。
本层选择不子类化，因为改游戏的窗口过程比读系统状态更容易破坏游戏。

### 1.2 由此产生的三个结构性事实

1. **每帧轮询，而不是事件驱动。** 所有输入都是"问一次当前状态"。
   这解释了为什么每个键都需要自己的边沿锁存（`m->key_down[]`、
   `menu_ui_*_was_down`）——没有事件就没有"按下"这件事，只能自己造。

2. **"是否在前台"必须显式门控，而且必须在每个读取点单独加。**
   没有统一的输入事件入口，所以没有统一的地方可以加过滤。**这是第 2 条评审
   缺陷的结构性原因**：`nr_layer.c` 加了门控，`nr_menu.c` 忘了加，
   两者互不知道。

3. **系统级读取意味着**一旦漏掉门控，影响范围是**整个桌面**而不是游戏。
   在聊天窗口按 `L` 会改游戏设置，在编辑器里按 Home 会关掉游戏里的效果。

---

## 2. 分层与职责

```
┌─────────────────────────────────────────────────────────────┐
│ nr_layer.c  (3667 行)                                        │
│   Vulkan 图层本体：hook vkQueuePresentKHR、抓帧、送 daemon、   │
│   合成、以及"什么时候该画菜单"的决策。                        │
│   对菜单有 89 行调用点 —— 既是最大消费者，也是耦合最重的地方    │
└───────────────┬─────────────────────────────────────────────┘
                │ 直接 C 调用（无 dispatch 层）
    ┌───────────┴───────────┬──────────────────┬───────────────┐
    ▼                       ▼                  ▼               ▼
┌────────────┐    ┌──────────────────┐  ┌──────────┐  ┌─────────────┐
│ nr_menu_ui │    │ nr_menu_bridge   │  │ nr_menu  │  │ nr_raw_mouse│
│  (C++)     │    │ (C, 1708 行)     │  │ (C,1724) │  │ (C, 180)    │
│ ImGui 面板 │    │ 旋钮表 + 文件读写 │  │ 手绘菜单 │  │ 原始输入    │
│ 1454 行    │    │ + pacing 管理     │  │ + 按键   │  │             │
└─────┬──────┘    └────────┬─────────┘  └────┬─────┘  └─────────────┘
      │                    │                 │
      │              ┌─────┴──────┐    ┌─────┴──────┐
      │              ▼            ▼    ▼            ▼
      │      ┌────────────┐  ┌──────────┐  ┌────────────┐
      │      │nr_menu_    │  │ nr_file  │  │  nr_focus  │
      │      │strings.c   │  │ (原子写) │  │ (32 行)    │
      │      │(生成, 4语) │  └──────────┘  └────────────┘
      │      └────────────┘
      ▼
┌────────────────┐
│ third_party/   │
│ imgui 1.93.0   │  (vendored, byte-identical to upstream)
└────────────────┘
```

### 2.1 两个 UI，一套数据

**手绘路径（`nr_menu.c`）** 和 **ImGui 路径（`nr_menu_ui.cpp`）** 并存。
这不是遗留，是**故意的降级路径**：

- `nr_menu_create` 会问物理设备"这个 swapchain 格式能不能做 storage image"，
  不能则返回 NULL（`nr_menu.h:28-31`）
- `nrex_create` 是 ImGui 那条，失败也返回 NULL
- `nr_layer.c` 对两者都按"NULL 就是没有菜单"处理

`nr_menu_set_key_reading(menu, 0)`（`nr_layer.c:2223`）在 ImGui 生效后
关掉手绘路径的按键读取，避免**两条路径对同一个键各做一次**。
这个机制的存在本身说明了两路径并存的成本——它是第 2 条缺陷的另一个成因：
按键读取分散在 `nr_layer.c`、`nr_menu.c`、`nr_menu_ui.cpp` 三处，
每处都要独立记得门控。

### 2.2 `nr_menu_bridge` 是唯一的数据权威

它拥有：
- 旋钮表（11 个，镜像 `nr_knobs.py`）
- 设置文件 `work/nr_settings.json` 的读写（唯一写入者）
- **实时预览** `nr_settings.json.live`（带 PID，daemon 据此判断作者是否存活）
- pacing 的三个值（存在 `nr-env.txt`，见 2.3）
- 语言、效果开关的持久化

**"Save 是唯一写入设置文件的动作"** 是这一层的核心规则。
所有其它路径只写预览。这条规则曾因 Reset 调用 `save_settings` 而被破坏过
（一次按键销毁用户全部保存值），修复后写进了注释与测试。

### 2.3 两套设置，两个文件，两个读者

| | 文件 | 读者 | 何时生效 |
|---|---|---|---|
| 画质旋钮 | `work/nr_settings.json` | daemon（每帧读） | 下一帧 |
| 帧节奏/GPU 路径 | 游戏旁 `dlss-nr\nr-env.txt` | **图层**（启动读一次） | 下次启动 |

这不是随意的划分：图层读环境变量的时机在 `vkCreateInstance`，
之后再也不读；daemon 是长驻进程，每帧看文件。所以"下次启动生效"
对前者是物理事实，不是设计选择。

**归属规则（第三轮确定）**：
- `nr-env.txt` 归**面板**——它是唯一能看见图层实际在跑什么的写入者
- 面板同时写 `windows-profile.json`，profile 是**投影**
- 窗口从 profile 读，两边因此一致

这条规则的破坏与恢复构成了第 6 条评审缺陷。

---

## 3. ABI 边界：唯一的 C++/C 交界

`nr_menu_ui.h` 是**唯一**跨 C++/C 的接口。它必须遵守两条：

### 3.1 `extern "C"`

`nr_menu_ui.h` 有（`#ifdef __cplusplus extern "C" {`）。
**`nr_menu_strings.h` 曾经没有**——它是生成的，生成器忘了加。
后果：C++ 调用方去找修饰名 `?nr_menu_text@@YAPEBDHH@Z`，
直到有 C++ 测试第一次调用它才暴露链接失败。

**教训**：生成的头文件和手写的头文件在 ABI 契约上必须同等对待。

### 3.2 调用约定（x86 才致命）

```c
// nr_menu_ui.cpp:122 —— 现状
typedef PFN_vkVoidFunction (*nrex_gpa_fn)(void *device, const char *name);
```

在 x64 上所有调用约定统一，这段代码是正确的。
在 x86 上 `vkGetDeviceProcAddr` 是 `__stdcall`，而这里声明成 `__cdecl`：
**每次解析函数指针都让栈不平衡 8 字节**。累积到一定程度就是
`c0000005` 跳到野地址。

正确写法是 `VKAPI_PTR`（`Vulkan` 的调用约定宏，在 32 位上展开为 `__stdcall`）。

**为什么这个缺陷只在 32 位上出现，而 32 位构建确实存在：**
`tools/build_win.bat:257` 用 `%VCX86%`（x86 vcvars）编译 `nr_menu_ui.cpp`，
`build_win.bat:261` 以 `/MACHINE:X86` 把它链接进 `nr_layer32.dll`。
所以 32 位 DLL **带着完整菜单**出厂，这条 ABI 错误随之下发。

### 3.3 句柄宽度（x86 第二个陷阱）

Vulkan 的非可分派句柄（`VkImage`、`VkImageView`、`VkBuffer`…）：

- `VK_USE_64_BIT_PTR_DEFINES` 为 1 时 → 真指针（x64）
- 为 0 时 → `uint64_t`（32 位）

`nr_menu_ui.cpp:260-270` 的注释已经意识到了（"pointers only when...
32-bit games"），并正确使用了 `VK_NULL_HANDLE`。
但 `nrex_image` / `nrex_image_view` **返回值**仍是 `void *`，
在 32 位上会截断 64 位句柄。

这就是作者的第一处怀疑修好之后仍然崩溃的原因——两个独立缺陷。

### 3.4 这个边界上值得保留的做法

- `VK_NO_PROTOTYPES` + 运行时解析：图层**不能**链接自己的 Vulkan 库，
  必须骑在游戏已加载的那一个上。这决定 `nrex_vk` 表的存在。
- `nrex_gpa_fn` 通过 `ctx[0]`/`ctx[1]`（`void *`）传递：在 x64 上安全，
  在 x86 上函数指针也是 32 位，安全。**这一段没问题**。

---

## 4. 输入子系统的三条路径

这是缺陷最集中的地方，值得单独画出。

```
                  ┌──────────────────────────────┐
                  │  系统级状态（无事件）          │
                  │  GetForegroundWindow          │
                  │  GetAsyncKeyState             │
                  │  GetCursorPos                 │
                  └───────────┬──────────────────┘
                              │
        ┌─────────────────────┼──────────────────────┐
        ▼                     ▼                      ▼
┌───────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ nr_layer.c    │   │ nr_menu.c        │   │ nr_menu_ui.cpp    │
│ menu_read_keys│   │ nr_menu_update   │   │ menu_ui_keys      │
│ (L2831+)      │   │ (L891+)          │   │ (L1260+)          │
│               │   │                  │   │                   │
│ INS  ✅门控   │   │ INS  ✅read_keys │   │ 所有键 ⚠️ 无门控  │
│ L    ✅门控   │   │ Home ❌ 无门控   │   │ 但由 layer 调用方 │
│ Home ✅门控   │   │ S/R  ❌ 只等打开 │   │ 前置判断 foreground│
│ S/R  ✅门控   │   │ 方向键 ✅read_keys│  │                   │
└───────────────┘   └──────────────────┘   └──────────────────┘
        │                     │                      │
        └─────────────────────┴──────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │ 结果：三种行为    │
                    │ 前台开/前台关/后台│
                    └───────────────────┘
```

### 4.1 门控必须加在哪

门控的正确位置是**读取点**，不是消费点。因为：
- 消费点（`nr_menu_inject_*`）不知道键从哪来
- 读取点是唯一知道"这是系统级状态"的地方

**但 `nr_menu.c` 拿不到"游戏窗口"是哪个窗口。**
它没有 `HWND`，只有 `nr_menu*`。所以正确的做法是：

**给 `nr_menu` 提供一个焦点查询能力**（函数指针或 hwnd），
由 `nr_layer.c` 在 `nr_menu_create` 时注入。
这样 `nr_menu.c` 能在读取点门控，而不需要自己猜窗口。

### 4.2 三条硬规则

1. **Home 面板开关都工作。** 这是产品需求，不是妥协。
   门控只看"游戏是否在前台"，不看"面板是否打开"。
2. **S/R 只在面板打开时工作**（面板本身就是"我在改设置"的声明）。
3. **L 需要面板打开 + 前台**（语言变更会写文件，属于设置动作）。

### 4.3 按键透传给游戏

**无法完全避免。** `GetAsyncKeyState` 只读，不从系统队列里取走事件。
游戏自己的消息循环照常收到按键。

可选的抑制手段及其代价：
- `RegisterHotKey`：全局独占，会破坏游戏自身的绑定，且需要窗口
- 低级键盘钩子（`WH_KEYBOARD_LL`）：能做到，但在别人进程里装全局钩子
  是这类图层最不该做的事——它影响整机输入，出错范围不可控

本层选择**不抑制**，并在面板上说明。这是有意的边界，不是遗漏。

---

## 5. 数据流：一次按键到一次落盘

以"在面板里把 intensity 从 1 调到 1.5，然后按 S"为例：

```
1. nr_layer.c menu_read_keys         GetAsyncKeyState 检测到方向键
2. nr_menu_inject_step / nrex_*      传给 UI
3. nr_layer.c:3203-3221              读 nrex 返回的 changed 行
4. nr_menu_bridge_set(i, 1.5)        ──┐
                                      │ 桥接层
5. format_row()                        │ 只改内存 + shown 字符串
6. write_live_settings()               │ 写 nr_settings.json.live（带 PID）
                                      ──┘
7. daemon 每帧读 .live，看到 PID 存活 → 应用 1.5 → 画面变化
                                      （此时 nr_settings.json 尚未变）

8. 用户按 S
9. nr_menu_take_save_request
10. nr_menu_bridge_save              ──┐
                                      │ 唯一写设置文件的路径
11. save_settings()                    │ 重写 nr_settings.json
12. preview_confirmed = 1              │ 预览被"确认"
13. write_live_settings()              │ 再写一次，两边一致
                                      ──┘
```

**关键不变式**：
- 第 4-6 步绝不碰 `nr_settings.json`
- 第 11 步是唯一写入
- `preview_confirmed` 是"这个预览能不能在下次启动被应用"的开关
  （Reset 后设为 0，Save/Revert 后设为 1）

### 5.1 pacing 走另一条路

```
nr_menu_bridge_set_pacing(what, value)
  ├── 冲突消解（GPU 路径要求异步关 + 每帧）
  ├── write_pacing()          → nr-env.txt（图层启动时读）
  └── write_profile_pacing()  → work/windows-profile.json（窗口读）
```

**注意 `NR_GPU_ONLY` 恒写 0**：它是严格模式，不是 GPU 路径的另一个名字。
写 1 会在共享路径不可用时拒绝 CPU 回退，让效果完全消失——这正是
issue #17 的形态。这条注释在 `nr_menu_bridge.c:495-504`，改动前必须读。

### 5.2 三个 pacing 值的联动

| 操作 | 副作用 | 原因 |
|---|---|---|
| 开异步 | 关 GPU 路径 | 路径要求同步 |
| 开 GPU 路径 | 关异步 + cadence=1 | 同上 |
| cadence>1 | 关 GPU 路径 | 同上 |

这些规则**只在一处实现**（`nr_menu_bridge.c:1518-1545`），
UI 两处都通过它。这是正确做法，值得保持。

---

## 6. 本地化：一个生成器，两个消费者

```
dist-tools/windows-wizard.ps1  (安装窗口的 4 语言表, 178 键)
            │
            │ tools/gen_menu_strings.py
            ▼
   ┌────────────────────┬────────────────────┐
   ▼                    ▼                    ▼
nr_menu_strings.c   nr_menu_strings.h   nr_menu_strings.py
(游戏内菜单用)      (声明，含 extern "C")  (Python 工具用)
```

**为什么生成而不是手写**：本项目已经因为"同一事实三份拷贝"出过两次错
（`render_scale` 默认值在 3 处不一致；`Profile.neutral` 在一个 UI 里有、
另一个 UI 里已删）。

### 6.1 菜单独有的键走 `OWN`

安装窗口没有 overlay，所以 `compare`、`menu key` 这类键在它那里不存在。
这些在 `gen_menu_strings.py` 的 `OWN` 字典里手写 4 种语言。

**第三轮新增的 pacing 相关键**（`async`、`async_hint`、`gpu_image`、
`gpu_image_hint`、`every`、`every_hint`、`every_frame`、`every_nth`、
`gpu_blocked_*`）也在这里——它们曾经硬编码在 `nr_layer.c` 里用
`zh ? ... : ...`，即**四语言枚举上做了二选一**，导致俄语和西班牙语
落到英文分支。

**教训**：任何 `== NR_LANG_ZH ? ... : ...` 都是缺陷的形状。四语言枚举
只能配查表，不能配三元。

### 6.2 复数形式是格式化，不是拼接

`NR_STR_EVERY_NTH` 用 `{}` 占位（如 `"every {} frames"`），
`nr_layer.c` 替换它。这样语序不同的语言不需要改 C 代码。

**但俄语需要三套形式**：1 кадр / 2-4 кадра / 5+ кадров。
当前只有一个 `every_nth` 键，且缓冲区 24 字节不够（"каждые 2 кадров" 27 字节）。
这是第 8 条评审的两半：
- **缓冲区**：`every_names_store[4][24]` → 至少 32
- **语法**：需要按数字选形式，不是简单替换

### 6.3 检查能覆盖什么，不能覆盖什么

| 检查类型 | 能发现 | 不能发现 |
|---|---|---|
| 键存在（自检的 178 键） | 缺失的翻译 | 内容过时 |
| 不等于英文 | 直接拷贝英文 | **翻译落后于英文修订** |
| `{}` 保留 | 格式化被破坏 | 语序正确性 |

**第三轮实际案例**：安装窗口的 `GpuImagePathTip` 中文写的是
"如果不满足则跳过效果，游戏显示原始帧"，而英/俄/西写的是
"走常规路径，保存报告说明原因"。前者是 issue #17 **修掉之前**的行为。
中文确实不等于英文（检查通不过它），但**内容是错的**。
这类问题只能靠并排读四种语言发现，没有自动办法。

**结论：本地化检查只能保证"没有漏"，不能保证"说得对"。**
后者必须人工审阅，且每次英文修订后都要重看其余三语。

---

## 7. 构建：同一份源码，五种构建方式

| 构建 | 目标 | 现状 |
|---|---|---|
| MSVC（build_win.bat） | `nr_layer.dll` / `nr_layer32.dll` + 7 个 fixture | ✅ 工作，链接菜单全部源 |
| MinGW（CMake on Windows） | 同上 + 测试 | ❌ 3 个测试 undefined refs |
| Linux（Makefile） | `libnr_layer.so` + 测试 | ❌ 未构建菜单源 |
| Linux（CMake） | 同上 | ❌ 同 Makefile |
| 交叉（-m32） | `nr_layer32` | ⚠️ 链接可能通，运行必崩（第 1 条） |

### 7.1 根因

五个测试用 **`#include "nr_layer.c"`** 而不是链接库：

`test_device_teardown.c:25`、`test_exchange.c:8`、`test_gpu_only.c:22`、
`test_settled.c:6`、`test_transfer_resources.c:3`

它们因此继承 `nr_layer.c` 对菜单的全部符号依赖，而构建系统只给它们
一个 `.c` 文件。**`test_gpu_present` 已经在用 `add_dependencies(... nr_layer ...)`，
说明链接库是既有模式，这五个只是没跟进。**

### 7.2 唯一能工作的构建做对了什么

`build_win.bat:150-168` 显式把 ImGui 五源 + `nr_menu_ui.cpp` +
`nr_menu.c`/`bridge`/`font`/`strings`/`strings_lookup`/`file`/`focus`/
`raw_mouse` 编成 obj，再链接进 `nr_layer.dll`。
**其它构建要复制的就是这份清单。**

### 7.3 平台门控的真实状况（不是注释说的那样）

`nr_layer.c` 的注释说菜单头文件"bring their own no-op arm"。
实际情况是**混合的**：

| 文件 | 有无 no-op arm |
|---|---|
| `nr_focus.h` | ✅ `static inline`（L45-46） |
| `nr_raw_mouse.c` | ✅ `#else` stub（L174-179） |
| `nr_menu_font.c` | ✅ `#else` stub glyph_bitmap（L106-115） |
| `nr_menu.c` | ✅ 按键函数有 `#else`（L282-297） |
| **`nr_menu_ui.cpp`** | ❌ **`menu_ui_keys` 整段无门控**（L1316-1339） |
| **`nr_menu.h`** | ❌ 声明无条件（但声明本身可编译） |

所以"Linux 能编译"这个前提**对 `nr_menu_ui.cpp` 不成立**。
这是第 5 条修复必须先做的工作：给那个文件加门控。

---

## 8. 这个架构里最容易犯的错

按实际发生次数排列：

1. **系统级读取忘了门控**（第 2 条；也是最早"拖另一个窗口的进度条动了滑块"的原因）
2. **同一事实存在多份拷贝**（`render_scale` 默认值、`Profile.neutral`）
3. **四语言枚举上做二选一**（第 8 条之前的 pacing 行）
4. **"改了一处以为改全了"**（`nr_layer.c` 门控了，`nr_menu.c` 没有）
5. **测试断言了实现而不是行为**（"Reset 不动文件"曾因 Reset 没跑而假绿）
6. **构建系统与源码脱节**（本次第 4、5 条）
7. **注释描述了不存在的东西**（"bring their own no-op arm"）

### 8.1 写在这里的用途

新加一个系统级读取点时，问：
- 它有没有门控？
- 门控用的"游戏窗口"是从哪来的？
- 非 Windows 上它是什么？

新加一个字符串时，问：
- 是不是 `zh ? ... : ...`？
- 四种语言都写了吗？
- 缓冲区够吗？

新加一个构建目标时，问：
- 它编 `nr_layer.c` 吗？那它需要菜单源吗？
- 是 `#include "nr_layer.c"` 还是链接库？

---

## 9. 未决问题

| 问题 | 影响 | 谁定 |
|---|---|---|
| 面板的"网络"框与 `weights` 旋钮重复（`9a74998` 加的） | 两个控件同名不同值 | 作者 + 所有者 |
| pacing 双向写入的具体断言 | 第 6 条的测试改法 | 先实现再与作者确认 |
| Linux 上是否需要菜单（还是永远 no-op） | 第 5 条的深度 | 作者（他说"stub the menu out off Windows"） |
| 面板能否提示"游戏仍会收到按键" | 第 7 条的语气与位置 | 实现时定 |

---

## 10. 一句话总结

**这一层是一个没有窗口、没有消息循环、没有事件的特权观察者。**
它所有的输入都是轮询系统状态，所有的绘制都是借用别人的 swapchain，
所有的配置都写在别人进程能读的两个文件里。
它的稳健性不来自精巧的设计，而来自**每个读取点都记得门控、
每份数据都只有一个权威、每种平台都显式处理**——
这三件事任何一件漏掉，症状都会出现在完全无关的地方。
