"""项目内的异常类型。命令行把 RoboArenaError 打印成一行中文报错，其余异常照常抛出。"""


class RoboArenaError(Exception):
    """可预期的、应该直接告诉使用者的错误。"""


class ConfigError(RoboArenaError):
    """配置文件或覆盖项有误。"""


class NotFoundError(RoboArenaError):
    """按名称或编号找不到对象。"""


class ConflictError(RoboArenaError):
    """对象已存在，或与已有记录冲突。"""


class IntegrityError(RoboArenaError):
    """文件内容与登记的哈希不一致。"""


class ModelError(RoboArenaError):
    """语言模型的服务调用失败：连不上、超时、服务端报错，或脚本化假模型的脚本用完了。

    retryable 表示再试一次可能成功（超时、连不上、服务端 5xx、限流）；请求本身有错时为 False。
    """

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class ToolError(RoboArenaError):
    """Agent 的工具调用不成立：工具不存在、参数没通过校验，或没有得到许可。"""
