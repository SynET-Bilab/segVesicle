import xml.etree.ElementTree as ET
import numpy as np
import os
from qtpy.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QGridLayout, QComboBox, QMessageBox
)
from qtpy.QtCore import Qt
import napari

class FixFNWindow(QMainWindow):
    def __init__(self, parent):
        super().__init__(parent.viewer.window.qt_viewer)
        self.setWindowTitle('Fix Flase Negative Vesicles')
        self.parent = parent
        self.viewer = parent.viewer
        self.current_index = 0  # Current vesicle start index

        # Filter vesicles based on ori_filter_xml_path and filter_xml_path
        vesicle_ids = self.filter_vesicle_ids()

        # Load target vesicles from class_xml_path based on filtered IDs
        self.vesicles = self.load_target_vesicles_from_xml(vesicle_ids)

        # Initialize class dictionary to save each vesicle's category
        self.classes = {vesicle.attrib['vesicleId']: 'others' for vesicle in self.vesicles}
        for vesicle in self.vesicles:
            vesicle_id = vesicle.attrib['vesicleId']
            type_element = vesicle.find('Type')
            if type_element is not None:
                self.classes[vesicle_id] = type_element.attrib.get('t', 'others')

        # Create 12 Napari Viewer instances to display image patches
        self.patch_viewers = [napari.Viewer(show=False) for _ in range(12)]
        self.patch_layers = [None] * 12

        # Save each category input box for later operations
        self.class_inputs = [None] * 11  # First 11 correspond to category input boxes

        # Create UI elements
        self.init_ui()

        # Display the current vesicle group
        self.display_current_vesicles()

    def verify_files_exist(self):
        # Check if the required XML paths exist
        self.class_xml_path = self.parent.tomo_path_and_stage.class_xml_path
        self.ori_filter_xml_path = self.parent.tomo_path_and_stage.ori_filter_xml_path
        self.filter_xml_path = self.parent.tomo_path_and_stage.filter_xml_path
        return all(os.path.exists(path) for path in [self.class_xml_path, self.ori_filter_xml_path, self.filter_xml_path])

    def close_with_message(self, message):
        QMessageBox.information(self, "Notice", message)
        self.close()

    def filter_vesicle_ids(self):
        # Load ori_filter_xml and filter_xml files
        ori_tree = ET.parse(self.ori_filter_xml_path)
        ori_root = ori_tree.getroot()
        filter_tree = ET.parse(self.filter_xml_path)
        filter_root = filter_tree.getroot()

        # Find vesicle IDs that are Type='vesicle' in ori_filter_xml and Type='other' in filter_xml
        ori_vesicles = {vesicle.attrib['vesicleId']: vesicle.find('Type').attrib.get('t')
                        for vesicle in ori_root.findall('Vesicle') if vesicle.find('Type')}
        filter_vesicles = {vesicle.attrib['vesicleId']: vesicle.find('Type').attrib.get('t')
                           for vesicle in filter_root.findall('Vesicle') if vesicle.find('Type')}

        # Identify IDs to be annotated
        vesicle_ids = [vesicle_id for vesicle_id, v_type in ori_vesicles.items()
                       if v_type == 'vesicle' and filter_vesicles.get(vesicle_id) == 'other']
        return vesicle_ids

    def load_target_vesicles_from_xml(self, vesicle_ids):
        # Load class_xml file and retrieve vesicles based on vesicle_ids
        class_tree = ET.parse(self.class_xml_path)
        class_root = class_tree.getroot()
        all_vesicles = class_root.findall('Vesicle')

        # Filter vesicles in class_xml that match the target IDs
        target_vesicles = [vesicle for vesicle in all_vesicles if vesicle.attrib['vesicleId'] in vesicle_ids]

        self.tree = class_tree
        self.root = class_root
        self.all_vesicles = all_vesicles  # All vesicles including annotated and unannotated

        return target_vesicles

    def init_ui(self):
        # Main widget
        main_widget = QWidget()
        main_layout = QGridLayout()

        # Define category options
        self.class_options = ['false', 'tether', 'contact', 'omega', 'pit', 'CCV', 'endosome', 'DCV', 'others']

        # Create a 3x4 grid layout
        for i in range(12):
            row = i // 4
            col = i % 4
            if i == 11:  # Last grid contains navigation buttons
                button_layout = QHBoxLayout()
                self.prev_button = QPushButton("Previous")
                self.prev_button.clicked.connect(self.prev_vesicles)
                self.next_button = QPushButton("Next")
                self.next_button.clicked.connect(self.next_vesicles)
                button_layout.addWidget(self.prev_button)
                button_layout.addWidget(self.next_button)
                container = QWidget()
                container.setLayout(button_layout)
                main_layout.addWidget(container, row, col)
            else:
                container = QWidget()
                container_layout = QVBoxLayout()

                # Add Napari Viewer
                container_layout.addWidget(self.patch_viewers[i].window.qt_viewer)

                # Add category dropdown
                vesicle_class_input = QComboBox()
                vesicle_class_input.addItems(self.class_options)
                vesicle_class_input.setCurrentText('others')
                vesicle_class_input.setObjectName(f'class_input_{i}')
                vesicle_class_input.currentTextChanged.connect(lambda val, idx=i: self.update_class(idx, val))
                container_layout.addWidget(vesicle_class_input)

                container.setLayout(container_layout)
                main_layout.addWidget(container, row, col)
                self.class_inputs[i] = vesicle_class_input  # Save reference to input box

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def display_current_vesicles(self):
        total_vesicles = len(self.vesicles)
        for i in range(11):  # First 11 for displaying vesicles
            idx = self.current_index + i
            if idx < total_vesicles:
                vesicle = self.vesicles[idx]
                center = vesicle.find('Center')
                x = float(center.attrib['X'])
                y = float(center.attrib['Y'])
                z = float(center.attrib['Z'])
                vesicle_id = vesicle.attrib['vesicleId']

                # Update category dropdown value
                class_value = self.classes[vesicle_id]
                vesicle_class_input = self.class_inputs[i]
                if vesicle_class_input:
                    vesicle_class_input.setEnabled(True)
                    vesicle_class_input.setCurrentText(class_value if class_value in self.class_options else 'others')

                patch = self.get_patch_around_point(z, y, x, size=128)

                # Adjust grayscale
                min_val = np.percentile(patch, 0.1)
                max_val = np.percentile(patch, 99)

                # Update Napari Viewer
                if self.patch_layers[i]:
                    self.patch_viewers[i].layers.remove(self.patch_layers[i])
                self.patch_layers[i] = self.patch_viewers[i].add_image(
                    patch, name=f'Vesicle {vesicle_id}',
                    contrast_limits=[min_val, max_val],
                    opacity=0.8,
                    gamma=0.75
                )
            else:
                # If no more vesicles, clear viewer and disable dropdown
                if self.patch_layers[i]:
                    self.patch_viewers[i].layers.clear()
                    self.patch_layers[i] = None
                vesicle_class_input = self.class_inputs[i]
                if vesicle_class_input:
                    vesicle_class_input.setEnabled(False)
                    vesicle_class_input.setCurrentText('others')  # Reset to default

    def save_current_classes(self):
        for i in range(11):
            idx = self.current_index + i
            if idx < len(self.vesicles):
                vesicle = self.vesicles[idx]
                vesicle_id = vesicle.attrib['vesicleId']
                class_value = self.classes[vesicle_id]
                # Update Type t attribute in XML
                type_element = vesicle.find('Type')
                if type_element is not None:
                    type_element.set('t', class_value)
                else:
                    # Create Type element if it doesn't exist
                    type_element = ET.SubElement(vesicle, 'Type')
                    type_element.set('t', class_value)

        # Save XML file to class_xml_path
        self.tree.write(self.class_xml_path, encoding='utf-8', xml_declaration=False)
        with open(self.class_xml_path, 'r+', encoding='utf-8') as file:
            content = file.read()
            file.seek(0)
            file.write(content.replace('"><', '">\n<'))

    def closeEvent(self, event):
        # Check if 'vesicles' is defined before attempting to save
        if hasattr(self, 'vesicles'):
            self.save_current_classes()
        super().closeEvent(event)
        
    
