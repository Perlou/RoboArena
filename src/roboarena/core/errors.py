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
