import json
import numpy as np
from scipy.spatial import KDTree
import napari
from napari.layers import Labels
from PyQt5.QtWidgets import (
    QVBoxLayout,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QPushButton,
    QHBoxLayout,
    QApplication,
    QMainWindow,
)
import sys

class VesicleViewer(QMainWindow):
    def __init__(self, main_viewer, json_file, label_layer_name='label', crop_size=64):
        """
        Initialize the VesicleViewer.

        Parameters:
        - main_viewer: The main Napari viewer instance
        - json_file: Path to the JSON file containing vesicle data
        - label_layer_name: Name of the label layer in the main viewer
        - crop_size: Size of the cropped cube (default is 64)
        """
        super().__init__()
        self.main_viewer = main_viewer
        self.json_file = json_file
        self.crop_size = crop_size
        self.vesicles, self.tree = self.get_info_from_json(json_file)
        self.label_data = self.main_viewer.layers[label_layer_name].data
        self.current_index = 0  # Index of the current vesicle being displayed

        # Set up the window
        self.setWindowTitle("Single Vesicle Display")

        # Create central widget and main layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # Create the left layout (Napari Viewer and navigation buttons)
        left_layout = QVBoxLayout()

        # Create an embedded Napari viewer
        self.viewer_window = napari.Viewer(ndisplay=3)
        left_layout.addWidget(self.viewer_window.window.qt_viewer)

        # Create navigation buttons
        nav_layout = QHBoxLayout()
        self.left_button = QPushButton("←")
        self.right_button = QPushButton("→")
        nav_layout.addWidget(self.left_button)
        nav_layout.addWidget(self.right_button)
        left_layout.addLayout(nav_layout)

        # Connect buttons to navigation functions
        self.left_button.clicked.connect(self.show_previous_vesicle)
        self.right_button.clicked.connect(self.show_next_vesicle)

        # Create the right layout (Table for vesicle info)
        right_layout = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(2)
        self.table.setHorizontalHeaderLabels(['Field', 'Value'])
        right_layout.addWidget(self.table)

        # Add left and right layouts to the main layout
        main_layout.addLayout(left_layout, stretch=3)
        main_layout.addLayout(right_layout, stretch=1)

        # Display the first vesicle if available
        if self.vesicles:
            self.display_vesicle(self.current_index)
        else:
            print("No vesicle data to display.")

    def get_info_from_json(self, json_file):
        """
        Read the JSON file and return vesicle information and a KDTree.

        Parameters:
        - json_file: Path to the JSON file

        Returns:
        - vesicles: List of vesicle information
        - tree: KDTree instance for spatial queries (if needed)
        """
        with open(json_file, "r") as f:
            vesicles = json.load(f)['vesicles']
        centers = [vesicle['center'] for vesicle in vesicles]
        centers = np.asarray(centers)

        if centers.size == 0:
            centers = np.empty((0, 3))
            tree = KDTree(np.empty((0, 3)))  # Create an empty KDTree
        else:
            tree = KDTree(centers, leafsize=2)

        return vesicles, tree

    def display_vesicle(self, index):
        """
        Display the vesicle at the given index.

        Parameters:
        - index: The index of the vesicle to display
        """
        vesicle = self.vesicles[index]
        center = np.array(vesicle['center'])  # (z, y, x)
        half_size = self.crop_size // 2

        # Define cropping boundaries and handle edge cases
        z_min = int(max(center[0] - half_size, 0))
        z_max = int(min(center[0] + half_size, self.label_data.shape[0]))
        y_min = int(max(center[1] - half_size, 0))
        y_max = int(min(center[1] + half_size, self.label_data.shape[1]))
        x_min = int(max(center[2] - half_size, 0))
        x_max = int(min(center[2] + half_size, self.label_data.shape[2]))

        # Crop label data for the selected vesicle
        cropped_label = self.label_data[z_min:z_max, y_min:y_max, x_min:x_max].copy()

        # Assume label values correspond to vesicle index +1
        label_value = index + 1

        # Keep only the target vesicle, set others to 0
        cropped_label[cropped_label != label_value] = 0
        cropped_label[cropped_label == label_value] = 1  # Convert to binary mask

        # Clear previous label layers in the viewer
        self.viewer_window.layers.selection.clear()
        for layer in self.viewer_window.layers:
            if isinstance(layer, Labels):
                self.viewer_window.layers.remove(layer)

        # Add the new label layer for the vesicle
        self.viewer_window.add_labels(cropped_label, name=vesicle['name'])

        # Center the camera on the vesicle
        self.viewer_window.camera.center = (half_size, half_size, half_size)

        # Update the table with vesicle information
        self.update_table(vesicle)

    def update_table(self, vesicle):
        """
        Update the table to display information about the selected vesicle.

        Parameters:
        - vesicle: Dictionary containing vesicle data
        """
        info = vesicle
        self.table.setRowCount(len(info))
        for i, (key, value) in enumerate(info.items()):
            # If value is a float, round to 2 decimal places
            if isinstance(value, float):
                value = f"{value:.2f}"
            elif isinstance(value, list):
                # For lists, round floats within the list to 2 decimal places
                value = [f"{v:.2f}" if isinstance(v, float) else v for v in value]
            self.table.setItem(i, 0, QTableWidgetItem(str(key)))
            self.table.setItem(i, 1, QTableWidgetItem(str(value)))
        self.table.resizeColumnsToContents()

    def show_previous_vesicle(self):
        """
        Display the previous vesicle in the list.
        """
        if not self.vesicles:
            return
        self.current_index = (self.current_index - 1) % len(self.vesicles)
        self.display_vesicle(self.current_index)

    def show_next_vesicle(self):
        """
        Display the next vesicle in the list.
        """
        if not self.vesicles:
            return
        self.current_index = (self.current_index + 1) % len(self.vesicles)
        self.display_vesicle(self.current_index)