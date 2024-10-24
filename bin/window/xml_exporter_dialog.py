import os
import xml.etree.ElementTree as ET
from PyQt5 import QtWidgets

def export_final_xml(main_viewer, tomo_path_and_stage, print_func):
    dialog = QtWidgets.QDialog(main_viewer)
    dialog.setWindowTitle("Enter Original Pixel Size")
    layout = QtWidgets.QVBoxLayout()

    input_layout = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel("Original Pixel Size (Å):")
    input_field = QtWidgets.QLineEdit()
    input_layout.addWidget(label)
    input_layout.addWidget(input_field)
    layout.addLayout(input_layout)

    apply_button = QtWidgets.QPushButton("Apply")
    layout.addWidget(apply_button)

    dialog.setLayout(layout)

    def on_apply():
        ori_pixel_size_a = input_field.text()
        try:
            ori_pixel_size_a = float(ori_pixel_size_a)
            if ori_pixel_size_a <= 0:
                raise ValueError("Pixel size must be a positive number.")
        except ValueError as e:
            print_func(f"Invalid input: {e}")
            return

        ori_pixel_size_nm = ori_pixel_size_a / 10

        if not os.path.exists(tomo_path_and_stage.class_xml_path):
            print_func(f"XML file does not exist: {tomo_path_and_stage.class_xml_path}")
            return

        try:
            tree = ET.parse(tomo_path_and_stage.class_xml_path)
            root = tree.getroot()
        except ET.ParseError as e:
            print_func(f"Failed to parse XML file: {e}")
            return

        current_pixel_size_str = root.attrib.get("pixelSize")
        if current_pixel_size_str is None:
            print_func("XML file is missing the pixelSize attribute.")
            return
        try:
            current_pixel_size_nm = float(current_pixel_size_str)
        except ValueError:
            print_func("pixelSize attribute is not a valid number.")
            return

        scale =  current_pixel_size_nm / ori_pixel_size_nm
        vesicle_count = 0
        for vesicle in root.findall('Vesicle'):
            type_element = vesicle.find('Type')
            # 检查是否存在 'Type' 元素及其 't' 属性
            if type_element is not None and type_element.attrib.get('t') != 'false':
                vesicle_count += 1
            scale_coordinates(vesicle, 'Center', scale, print_func)
            scale_coordinates(vesicle, 'Center2D', scale, print_func)
            scale_coordinates(vesicle, 'Center3D', scale, print_func)
            scale_radius(vesicle, 'Radius', scale, print_func)
            scale_radius(vesicle, 'Radius2D', scale, print_func)
            scale_radius(vesicle, 'Radius3D', scale, print_func)
            scale_distance(vesicle, 'Distance', scale, print_func)
            scale_coordinates(vesicle, 'ProjectionPoint', scale, print_func)
            scale_coordinates(vesicle, 'PitPoint', scale, print_func)

        root.set('pixelSize', f"{ori_pixel_size_nm}")
        root.set('vesicleCount', str(vesicle_count))

        try:
            write_xml_without_declaration(tree, tomo_path_and_stage.final_xml_path)
        except Exception as e:
            print_func(f"Failed to write final XML file: {e}")
            return

        for type_t, path in [('tether', tomo_path_and_stage.tether_xml_path), 
                             ('contact', tomo_path_and_stage.contact_xml_path), 
                             ('omega', tomo_path_and_stage.omega_xml_path)]:
            root_filtered = filter_vesicles(root, type_t)
            filtered_tree = ET.ElementTree(root_filtered)
            try:
                write_xml_without_declaration(filtered_tree, path)
            except Exception as e:
                print_func(f"Failed to write {type_t} XML file: {e}")
                return

        print_func("XML files successfully resampled and saved.")
        dialog.accept()

    apply_button.clicked.connect(on_apply)
    dialog.exec_()

def scale_coordinates(vesicle, element_name, scale, print_func):
    element = vesicle.find(element_name)
    if element is not None:
        for coord in ['X', 'Y', 'Z']:
            val = element.attrib.get(coord)
            if val is not None:
                try:
                    scaled_val = float(val) * scale
                    element.set(coord, f"{scaled_val}")
                except ValueError:
                    print_func(f"Failed to scale {element_name} coordinate: {coord}")

def scale_radius(vesicle, element_name, scale, print_func):
    element = vesicle.find(element_name)
    if element is not None:
        for r_attr in ['r', 'r1', 'r2', 'r3']:
            r = element.attrib.get(r_attr)
            if r is not None:
                try:
                    scaled_r = float(r) * scale
                    element.set(r_attr, f"{scaled_r}")
                except ValueError:
                    print_func(f"Failed to scale {element_name}: {r_attr}")

def scale_distance(vesicle, element_name, scale, print_func):
    element = vesicle.find(element_name)
    if element is not None:
        d = element.attrib.get('d')
        if d is not None:
            try:
                scaled_d = float(d) * scale
                element.set('d', f"{scaled_d}")
            except ValueError:
                print_func(f"Failed to scale {element_name}")

def filter_vesicles(root, type_t):
    new_root = ET.Element(root.tag, root.attrib)
    vesicle_count = 0
    for vesicle in root.findall('Vesicle'):
        type_element = vesicle.find('Type')
        if type_element is not None and type_element.attrib.get('t') == type_t:
            vesicle_count += 1
            new_root.append(vesicle)
    new_root.set('vesicleCount', str(vesicle_count))
    return new_root

def write_xml_without_declaration(tree, path):
    """Write XML tree to file without the XML declaration."""
    with open(path, 'wb') as f:
        tree.write(f, encoding='utf-8', xml_declaration=False)
    with open(path, 'r') as f:
        content = f.read()
    content = content.replace('><Vesicle', '>\n<Vesicle')
    with open(path, 'w') as f:
        f.write(content)
