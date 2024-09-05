from qtpy.QtWidgets import QWidget, QVBoxLayout, QPushButton, QTabWidget


class ToolbarWidget(QWidget):
    def __init__(self):
        super().__init__()

        # Set minimum height
        self.setMinimumHeight(300) 

        # Create the tab widget
        self.tabs = QTabWidget()

        # Create tabs
        self.isonet_tab = QWidget()
        self.predict_tab = QWidget()
        self.memb_tab = QWidget()

        # Add tabs to the tab widget
        self.tabs.addTab(self.isonet_tab, "IsoNet")
        self.tabs.addTab(self.predict_tab, "Vesicle Predict")
        self.tabs.addTab(self.memb_tab, "Membrane Segment")
        
        # Create layout for IsoNet tab and add buttons
        self.isonet_layout = QVBoxLayout()
        self.isonet_tab.setLayout(self.isonet_layout)
        self.open_ori_image_button = self.add_button(self.isonet_layout, "Open Original Tomo")
        self.deconvolution_button = self.add_button(self.isonet_layout, "Deconvolution")
        self.correction_button = self.add_button(self.isonet_layout, "Correction")
        self.finish_isonet_button = self.add_button(self.isonet_layout, "Finish IsoNet")

        # Create layout for Vesicle Predict tab and add buttons
        self.predict_layout = QVBoxLayout()
        self.predict_tab.setLayout(self.predict_layout)
        self.draw_tomo_area_button = self.add_button(self.predict_layout, "Draw Tomo Area")
        self.predict_button = self.add_button(self.predict_layout, "Predict")
        self.manual_annotation_button = self.add_button(self.predict_layout, "Manual Annotation Only")

        # Create layout for the Memb tab and add buttons
        self.memb_layout = QVBoxLayout()
        self.memb_tab.setLayout(self.memb_layout)
        self.draw_memb_button = self.add_button(self.memb_layout, "Draw Membrane Area Mod")
        self.stsyseg_button = self.add_button(self.memb_layout, "Stsyseg")
        self.visualize_button = self.add_button(self.memb_layout, "Visualize")

        # Set the main layout for the widget
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tabs)
        self.setLayout(self.main_layout)

    def add_button(self, layout, name):
        button = QPushButton(name)
        layout.addWidget(button)
        return button