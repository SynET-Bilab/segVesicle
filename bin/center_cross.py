from napari import Viewer
from napari.utils.action_manager import action_manager
from napari.utils.notifications import show_info
import numpy as np
import napari
from napari.components.viewer_model import ViewerModel

# 函数中心十字移动到鼠标位置
def center_cross_on_mouse(viewer_model: ViewerModel):
    """move the cross to the mouse position"""
    try:
        if not getattr(viewer_model, "mouse_over_canvas", True):
            show_info("Mouse is not over the canvas. You may need to click on the canvas.")
            return

        viewer_model.dims.current_step = tuple(
            np.round(
                [
                    max(min_, min(p, max_)) / step
                    for p, (min_, max_, step) in zip(
                        viewer_model.cursor.position, viewer_model.dims.range
                    )
                ]
            ).astype(int)
        )
    except Exception as e:
        show_info(f"An error occurred: {e}")
        

action_manager.register_action(
    name='napari:move_point',
    command=center_cross_on_mouse,
    description='Move dims point to mouse position',
    keymapprovider=ViewerModel,
)

action_manager.bind_shortcut('napari:move_point', 'C')

action_manager._debug(True)