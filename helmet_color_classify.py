"""这个文件是兼容入口：命令行和 PowerShell 脚本仍然可以直接调用它，而真正的头盔颜色识别逻辑已经拆分到 `helmet_classifier/` 包中，分别按参数解析、头盔区域估计、颜色判定、中文绘制、调试输出和视频处理流程进行管理，方便后续维护和扩展。"""

from helmet_classifier.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
