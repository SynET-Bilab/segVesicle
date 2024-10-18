import os
import numpy as np
import xml.etree.ElementTree as ET
from qtpy.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton
from qtpy.QtGui import QDoubleValidator

class DistanceFilterWindow(QDialog):
    def __init__(self, tomo_viewer):
        super().__init__(tomo_viewer.viewer.window.qt_viewer)
        self.tomo_viewer = tomo_viewer
        self.setWindowTitle("Filter Vesicles by Distance")
        self.setModal(True)
        self.resize(300, 150)
        
        layout = QVBoxLayout()
        
        # Use QHBoxLayout to arrange the label and input field horizontally
        input_layout = QHBoxLayout()
        
        self.distance_label = QLabel("Distance (nm):")
        self.distance_input = QLineEdit()
        self.distance_input.setPlaceholderText("Enter distance in nm")
        self.distance_input.setValidator(QDoubleValidator(0.0, 1e6, 2))
        
        input_layout.addWidget(self.distance_label)
        input_layout.addWidget(self.distance_input)
        
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_filter)
        
        layout.addLayout(input_layout)  # Add the horizontal layout to the main layout
        layout.addWidget(self.apply_button)
        
        self.setLayout(layout)
    
    def apply_filter(self):
        """Apply vesicle filtering based on the input distance."""
        try:
            distance_nm = float(self.distance_input.text())
            if distance_nm <= 0:
                raise ValueError("Distance must be positive.")
            self.accept()
            self.filter_vesicle(distance_nm)
        except ValueError as e:
            # Print error message
            self.tomo_viewer.print(f"Invalid Input: {str(e)}")

    def get_pixel_size_from_xml(self, xml_path):
        """Extract the pixel size from the XML file."""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            pixel_size = float(root.attrib.get('pixelSize', 1.714))  # Default pixel size is 1.714
            return pixel_size
        except Exception as e:
            # Print error message
            self.tomo_viewer.print(f"Failed to get pixel size from XML: {str(e)}")
            return 1.714

    def filter_vesicle(self, distance_nm):
        """Filter vesicles based on distance and update the label data."""
        try:
            # Read the label data (assumed to be a 3D numpy array)
            label_data = self.tomo_viewer.viewer.layers['label'].data
            
            # Get the pixel size from the XML file
            pixel_size = self.get_pixel_size_from_xml(self.tomo_viewer.tomo_path_and_stage.ori_xml_path)
            
            # Parse the XML data
            tree = ET.parse(self.tomo_viewer.tomo_path_and_stage.ori_xml_path)
            root = tree.getroot()
            
            for vesicle in root.findall('Vesicle'):
                distance = float(vesicle.find('Distance').attrib['d'])
                if distance * pixel_size < distance_nm:
                    # Change Type to 'others'
                    type_elem = vesicle.find('Type')
                    if type_elem is not None and type_elem.attrib.get('t') == 'vesicle':
                        type_elem.set('t', 'others')
            
            # Save the modified XML
            tree.write(self.tomo_viewer.tomo_path_and_stage.filter_xml_path, encoding='utf-8', xml_declaration=True)
            
            # Parse the modified XML again
            tree_new = ET.parse(self.tomo_viewer.tomo_path_and_stage.filter_xml_path)
            root_new = tree_new.getroot()
            
            # Extract vesicle IDs marked as 'others'
            others_ids = [int(vesicle.attrib['vesicleId']) for vesicle in root_new.findall('Vesicle') if vesicle.find('Type').attrib.get('t') == 'others']
            
            # Update the label data, retaining only vesicles of type 'others'
            new_label_data = np.where(np.isin(label_data, others_ids), label_data, 0)
            
            # Add the updated labels to the viewer
            self.tomo_viewer.viewer.layers['label'].visible = False
            
            self.tomo_viewer.viewer.add_labels(new_label_data, name="filter_labels")
            self.tomo_viewer.viewer.layers["filter_labels"].opacity = 0.5
            
            # Print success message
            self.tomo_viewer.print(f"Filtering completed, new XML saved at {self.tomo_viewer.tomo_path_and_stage.filter_xml_path}")
        
        except Exception as e:
            # Print error message
            self.tomo_viewer.print(f"Vesicle filtering failed: {str(e)}")
