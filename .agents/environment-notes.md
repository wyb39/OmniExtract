# 环境访问提示

项目虚拟环境位于 `.venv\`，但它引用宿主机解释器：

`C:\Users\13603\AppData\Local\Programs\Python\Python310\python.exe`

默认沙盒无法访问该宿主机路径，因此直接运行
`.venv\Scripts\python.exe` 可能提示找不到 Python 或访问被拒绝。这是沙盒
隔离导致的，并不代表 `.venv` 不存在。

运行项目、测试或安装依赖时，应申请宿主侧执行权限。完整说明见
`.agents/ENVIRONMENT_ACCESS_NOTE.md`。
