# GitHub 同步指南

本文档说明如何将 PolarQuant 项目同步到 GitHub。

## 方法 1: 使用自动脚本（推荐）

### 1. 运行设置脚本

```bash
cd /Users/moyong/project/ai/turboquant/kimi25/polarquant
chmod +x scripts/setup_github.sh
./scripts/setup_github.sh
```

### 2. 按照提示操作

脚本会：
- 检查 GitHub CLI 是否安装
- 引导登录 GitHub（如未登录）
- 询问仓库名称（默认: polarquant）
- 询问是否私有仓库
- 自动创建仓库并推送代码

## 方法 2: 手动设置

### 步骤 1: 在 GitHub 创建仓库

1. 访问 https://github.com/new
2. 输入仓库名称: `polarquant`
3. 选择公开或私有
4. **不要** 初始化 README（已有）
5. 点击 "Create repository"

### 步骤 2: 添加远程仓库

```bash
cd /Users/moyong/project/ai/turboquant/kimi25/polarquant

# 添加远程仓库（替换 yourusername 为你的 GitHub 用户名）
git remote add origin https://github.com/yourusername/polarquant.git

# 验证
git remote -v
```

### 步骤 3: 推送代码

```bash
# 确保在 main 分支
git branch -M main

# 推送到 GitHub
git push -u origin main
```

### 步骤 4: 验证

访问 `https://github.com/yourusername/polarquant` 查看代码是否已上传。

## 方法 3: 使用 Makefile

```bash
# 1. 先手动在 GitHub 创建仓库
# 2. 然后运行:

make git-submit msg="Initial commit: PolarQuant with Chinese docs"
```

**注意**: 需要先在 Makefile 中设置 GitHub 用户名，或手动添加 remote。

## 常见问题

### Q: 没有 GitHub 账号
A: 访问 https://github.com/signup 注册

### Q: 提示 "Permission denied"
A: 需要配置 SSH 密钥或使用 HTTPS + Token

```bash
# 使用 SSH（推荐）
git remote add origin git@github.com:yourusername/polarquant.git

# 或使用 HTTPS + Token
git remote add origin https://TOKEN@github.com/yourusername/polarquant.git
```

### Q: 推送失败 "rejected"
A: 如果远程仓库有冲突，先拉取再推送：

```bash
git pull origin main --rebase
git push origin main
```

### Q: 如何更新已推送的代码

```bash
# 添加修改
git add .

# 提交
git commit -m "描述你的修改"

# 推送
git push origin main
```

或使用 Makefile:
```bash
make git-submit msg="描述你的修改"
```

## 推送后配置

### 添加仓库描述

在 GitHub 仓库页面点击 "Edit" 添加：

- **Description**: PolarQuant - 基于极坐标变换的无损量化算法
- **Website**: （可选，可添加文档链接）
- **Topics**: `quantization`, `compression`, `llm`, `kv-cache`, `machine-learning`

### 启用 GitHub Pages（可选）

如需托管文档：

1. 进入 Settings > Pages
2. Source 选择 "Deploy from a branch"
3. Branch 选择 "main"，文件夹选择 "/docs"
4. 点击 Save

## 项目文件清单

推送后 GitHub 仓库将包含：

```
polarquant/
├── .gitignore              # Git 忽略文件
├── Makefile               # 构建自动化（中文注释）
├── README.md              # 项目说明（中英双语）
├── pyproject.toml         # 包配置
├── GITHUB_SETUP.md        # 本文件
├── scripts/
│   └── setup_github.sh    # GitHub 设置脚本
├── polarquant/            # 主包
│   ├── __init__.py
│   ├── core.py           # 核心实现（中文注释）
│   └── utils.py          # 工具函数（中文注释）
├── tests/                 # 测试套件
│   ├── __init__.py
│   ├── test_utils.py     # 工具测试（中文注释）
│   ├── test_core.py      # 核心测试
│   └── test_integration.py  # 集成测试
├── examples/              # 示例代码
│   ├── basic_usage.py    # 基本用法
│   ├── kv_cache_demo.py  # KV Cache 演示
│   └── qwen_*.py         # Qwen 模型示例
└── docs/                  # 文档
    ├── README.md         # 文档说明
    ├── sequence_diagram.md  # 时序图（中文）
    ├── sequence_diagram.png # 时序图图片
    └── algorithm_flow.png   # 流程图
```

## 后续维护

### 日常更新流程

```bash
# 1. 修改代码
# ...

# 2. 测试
make test

# 3. 提交
git add .
git commit -m "修改描述"

# 4. 推送
git push origin main
```

### 版本发布

```bash
# 1. 更新版本号（在 pyproject.toml 中）

# 2. 打标签
git tag -a v0.1.0 -m "版本 0.1.0"

# 3. 推送标签
git push origin v0.1.0
```

## 获取帮助

- Git 文档: https://git-scm.com/doc
- GitHub 文档: https://docs.github.com
- 项目 Issues: （推送后在 GitHub 创建）
