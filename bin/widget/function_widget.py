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
        # self.analysis_tab = QWidget()
        self.vesicle_analysis_tab = QWidget()

        # Add tabs to the tab widget
        self.tabs.addTab(self.isonet_tab, "IsoNet")
        self.tabs.addTab(self.predict_tab, "Vesicle Predict")
        self.tabs.addTab(self.memb_tab, "Membrane Segment")
        # self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.vesicle_analysis_tab, "Vesicle Analysis")
        
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
        self.export_xlsx_button = self.add_button(self.predict_layout, "Export XLSX")

        # # Create layout for the Memb tab and add buttons
        self.memb_layout = QVBoxLayout()
        self.memb_tab.setLayout(self.memb_layout)
        self.draw_memb_button = self.add_button(self.memb_layout, "Draw Membrane Area Mod")
        self.stsyseg_button = self.add_button(self.memb_layout, "Stsyseg")
        self.visualize_button = self.add_button(self.memb_layout, "Visualize")
        self.manualy_draw_button = self.add_button(self.memb_layout, "Manually Draw")

        # Create layout for the Analysis tab and add buttons
        # self.analysis_layout = QVBoxLayout()
        # self.analysis_tab.setLayout(self.analysis_layout)
        # self.analyze_volume_button = self.add_button(self.analysis_layout, "Analyze by Volume")
        # self.analyze_distance_button = self.add_button(self.analysis_layout, "Analyze by Membrane Distance")
        # self.show_single_vesicle = self.add_button(self.analysis_layout, "Show Single Vesicle")

        # Create layout for the Vesicle Analysis tab and add buttons
        self.vesicle_analysis_layout = QVBoxLayout()  # New layout for Vesicle Analysis
        self.vesicle_analysis_tab.setLayout(self.vesicle_analysis_layout)
        self.distance_calc_button = self.add_button(self.vesicle_analysis_layout, "Calculate Vesicle to Membrane Distance")
        self.filter_by_distance_button = self.add_button(self.vesicle_analysis_layout, "Filter Vesicles by Distance")
        self.annotate_pit_button = self.add_button(self.vesicle_analysis_layout, "Annotate pit")
        self.annotate_vesicle_type_button = self.add_button(self.vesicle_analysis_layout, "Annotate Vesicle Type")
        self.multi_class_visualize_button = self.add_button(self.vesicle_analysis_layout, "Multi-class Visualization")
        self.export_final_xml_button = self.add_button(self.vesicle_analysis_layout, "Export Final Vesicle Data")

        # Set the main layout for the widget
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.tabs)
        self.setLayout(self.main_layout)

    def add_button(self, layout, name):
        button = QPushButton(name)
        layout.addWidget(button)
        return button