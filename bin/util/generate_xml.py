#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import xml.etree.ElementTree as ET

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
    tree.write(path, encoding='utf-8', xml_declaration=False)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Add line breaks for better readability
    content = content.replace('><Vesicle', '>\n<Vesicle')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def process_xml(class_xml_path, original_pixel_size_a, print_func=print):
    # Derive paths based on class_xml_path
    class_xml_dir = os.path.dirname(class_xml_path)
    base_filename = os.path.basename(class_xml_path)
    
    if not base_filename.endswith('_vesicle_class.xml'):
        print_func("Error: The class_xml_path should end with '_vesicle_class.xml'.")
        sys.exit(1)
    
    base_tomo_name = base_filename.replace('_vesicle_class.xml', '')
    final_xml_filename = f"{base_tomo_name}.xml"
    tether_xml_filename = f"{base_tomo_name}_tether.xml"
    contact_xml_filename = f"{base_tomo_name}_contact.xml"
    omega_xml_filename = f"{base_tomo_name}_omega.xml"
    
    final_xml_path = os.path.join(class_xml_dir, final_xml_filename)
    tether_xml_path = os.path.join(class_xml_dir, tether_xml_filename)
    contact_xml_path = os.path.join(class_xml_dir, contact_xml_filename)
    omega_xml_path = os.path.join(class_xml_dir, omega_xml_filename)
    
    # Convert original pixel size from Å to nm
    original_pixel_size_nm = original_pixel_size_a / 10
    
    if not os.path.exists(class_xml_path):
        print_func(f"Error: XML file does not exist: {class_xml_path}")
        sys.exit(1)
    
    try:
        tree = ET.parse(class_xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print_func(f"Error: Failed to parse XML file: {e}")
        sys.exit(1)
    
    current_pixel_size_str = root.attrib.get("pixelSize")
    if current_pixel_size_str is None:
        print_func("Error: XML file is missing the 'pixelSize' attribute.")
        sys.exit(1)
    try:
        current_pixel_size_nm = float(current_pixel_size_str)
    except ValueError:
        print_func("Error: 'pixelSize' attribute is not a valid number.")
        sys.exit(1)
    
    scale = current_pixel_size_nm / original_pixel_size_nm
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
    
    root.set('pixelSize', f"{original_pixel_size_nm}")
    root.set('vesicleCount', str(vesicle_count))
    
    try:
        write_xml_without_declaration(tree, final_xml_path)
    except Exception as e:
        print_func(f"Error: Failed to write final XML file: {e}")
        sys.exit(1)
    
    for type_t, path in [('tether', tether_xml_path), 
                         ('contact', contact_xml_path), 
                         ('omega', omega_xml_path)]:
        root_filtered = filter_vesicles(root, type_t)
        filtered_tree = ET.ElementTree(root_filtered)
        try:
            write_xml_without_declaration(filtered_tree, path)
        except Exception as e:
            print_func(f"Error: Failed to write '{type_t}' XML file: {e}")
            sys.exit(1)
    
    print_func("Success: XML files successfully resampled and saved.")
    print_func(f"Final XML: {final_xml_path}")
    print_func(f"Tether XML: {tether_xml_path}")
    print_func(f"Contact XML: {contact_xml_path}")
    print_func(f"Omega XML: {omega_xml_path}")

def main():
    parser = argparse.ArgumentParser(description="Process vesicle classification XML and generate scaled XML files.")
    parser.add_argument('class_xml_path', type=str, help="Path to the '_vesicle_class.xml' file.")
    parser.add_argument('original_pixel_size', type=float, help="Original Pixel Size in angstroms (Å).")
    
    args = parser.parse_args()
    
    if args.original_pixel_size <= 0:
        print("Error: Original Pixel Size must be a positive number.")
        sys.exit(1)
    
    process_xml(args.class_xml_path, args.original_pixel_size)

if __name__ == "__main__":
    main()
