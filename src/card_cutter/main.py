from __future__ import annotations

import sys

from card_cutter.ui.main_window import MainWindow, create_application


def main() -> int:
    application = create_application()
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())

