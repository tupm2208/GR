from PyQt5 import QtWidgets, QtCore
from src.UI.main_ui import MainUi
from src.main_stream import MainStream
import sys


def update_frame():
    global main_stream, ui, app, timer
    result = main_stream.process_stream()
    # if main_stream.counter_trackers.first_tracker is not None:
    #     print(main_stream.counter_trackers.first_tracker.get_identity())
    if result is None:
        timer.stop()
        sys.exit(app.exec_())
        return
    image, image2, face = result
    ui.update_main_image(image2)
    ui.update_main_face(face)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainUi()
    ui.setupUi(MainWindow)
    MainWindow.show()
    main_stream = MainStream()

    timer = QtCore.QTimer(MainWindow)
    timer.timeout.connect(update_frame)
    ui.set_timer(timer)
    ui.set_trackers(main_stream.trackers)
    ui.set_counter_trackers(main_stream.counter_trackers)
    timer.start(5)
    sys.exit(app.exec_())
