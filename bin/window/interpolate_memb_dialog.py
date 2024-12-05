import numpy as np
from qtpy.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QRadioButton,
    QLineEdit, QPushButton, QButtonGroup, QMessageBox
)


class InterpolateMembDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Interpolate Operation")

        layout = QVBoxLayout()

        # Select operation type
        self.option_label = QLabel("Please choose operation:")
        layout.addWidget(self.option_label)

        self.dilation_erosion_radio = QRadioButton("Dilation and Erosion")
        self.none_radio = QRadioButton("None")
        self.dilation_erosion_radio.setChecked(True)  # Default selection: Dilation & Erosion

        self.button_group = QButtonGroup()
        self.button_group.addButton(self.dilation_erosion_radio)
        self.button_group.addButton(self.none_radio)

        layout.addWidget(self.dilation_erosion_radio)
        layout.addWidget(self.none_radio)

        # Select membrane type
        self.membrane_label = QLabel("Please choose membrane type:")
        layout.addWidget(self.membrane_label)

        self.front_membrane_radio = QRadioButton("Front Membrane")
        self.rear_membrane_radio = QRadioButton("Rear Membrane")
        self.front_membrane_radio.setChecked(True)  # Default selection: Front Membrane

        self.membrane_group = QButtonGroup()
        self.membrane_group.addButton(self.front_membrane_radio)
        self.membrane_group.addButton(self.rear_membrane_radio)

        layout.addWidget(self.front_membrane_radio)
        layout.addWidget(self.rear_membrane_radio)

        # Input mask threshold
        self.threshold_label = QLabel("Mask Threshold:")
        layout.addWidget(self.threshold_label)

        self.threshold_input = QLineEdit("0.4")
        layout.addWidget(self.threshold_input)

        # Apply button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.accept)  # Connect to the dialog's accept method

        layout.addWidget(self.apply_button)

        self.setLayout(layout)

    def get_values(self):
        # Get selected operation
        operation = 'dilation_erosion' if self.dilation_erosion_radio.isChecked() else 'none'
        
        # Get selected membrane type
        if self.front_membrane_radio.isChecked():
            membrane_type = 'front'
        else:
            membrane_type = 'rear'

        # Get mask threshold
        try:
            threshold = float(self.threshold_input.text())
        except ValueError:
            threshold = 0.4  # Default value

        return operation, membrane_type, threshold