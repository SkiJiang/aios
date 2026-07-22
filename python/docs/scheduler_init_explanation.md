# `python/aios/scheduler/__init__.py` 源码解释报告

## 文件定位

`scheduler/__init__.py` 是 scheduler 子包的统一导出入口。

## 导出内容

```python
from .cache import CacheManager
from .decode import DecodeManager
from .prefill import PrefillManager
from .scheduler import Scheduler
from .table import TableManager
```

这些类共同构成 continuous batching 调度系统：

- `CacheManager`：分配物理 KV page 并写 page table。
- `DecodeManager`：管理运行中的 decode 请求集合。
- `PrefillManager`：管理 pending 请求和 prefill admission。
- `Scheduler`：总调度器。
- `TableManager`：管理请求槽位、page table 和 token pool。

## `__all__`

`__all__` 明确 scheduler 包的公共 API。

上层可以写：

```python
from aios.scheduler import CacheManager, Scheduler, TableManager
```

而不需要分别从子文件导入。

## 总结

`scheduler/__init__.py` 是调度子包的门面文件，集中导出 continuous batching 所需的核心管理器。
