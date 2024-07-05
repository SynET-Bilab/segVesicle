from qtpy.QtCore import QTimer
from napari import Viewer


class KeyBinder:
    def __init__(self, timer: QTimer, global_viewer: Viewer):
        self.timer = timer
        self.global_viewer = global_viewer

    def increment_dims_left(self):
        self.global_viewer.dims.set_current_step(0, self.global_viewer.dims.current_step[0] - 1)

    def increment_dims_right(self):
        self.global_viewer.dims.set_current_step(0, self.global_viewer.dims.current_step[0] + 1)

    def bind_keys(self):
        @self.global_viewer.bind_key('PageDown', overwrite=True)
        def hold_to_increment_left(viewer):
            """Hold to increment dims left in the viewer."""
            self.timer.timeout.connect(self.increment_dims_left)
            self.timer.start()
            yield
            self.timer.stop()
            self.timer.timeout.disconnect(self.increment_dims_left)

        @self.global_viewer.bind_key('PageUp', overwrite=True)
        def hold_to_increment_right(viewer):
            """Hold to increment dims right in the viewer."""
            self.timer.timeout.connect(self.increment_dims_right)
            self.timer.start()
            yield
            self.timer.stop()
            self.timer.timeout.disconnect(self.increment_dims_right)