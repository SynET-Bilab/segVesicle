import json
import fire
import pandas as pd

def json_to_excel(json_path, excel_path):
    # Load JSON data from file
    with open(json_path, 'r') as file:
        data = json.load(file)

    # Prepare data for DataFrame
    rows = []
    for vesicle in data["vesicles"]:
        row = {
            "Name": vesicle["name"],
            "Center X": vesicle["center"][2],
            "Center Y": vesicle["center"][1],
            "Center Z": vesicle["center"][0],
            "Radius 1": vesicle["radii"][0],
            "Radius 2": vesicle["radii"][1],
            "Radius 3": vesicle["radii"][2],
            "CCF": vesicle["CCF"],
            "Evec1_X": vesicle["evecs"][0][2],
            "Evec1_Y": vesicle["evecs"][0][1],
            "Evec1_Z": vesicle["evecs"][0][0],
            "Evec2_X": vesicle["evecs"][1][2],
            "Evec2_Y": vesicle["evecs"][1][1],
            "Evec2_Z": vesicle["evecs"][1][0],
            "Evec3_X": vesicle["evecs"][2][2],
            "Evec3_Y": vesicle["evecs"][2][1],
            "Evec3_Z": vesicle["evecs"][2][0]
        }
        rows.append(row)

    # Create DataFrame and save to Excel
    df = pd.DataFrame(rows)
    df.to_excel(excel_path, index=False)
    print(f"Excel file has been saved to: {excel_path}")
    

if __name__ == '__main__':
    # json_to_excel(json_path='/home/liushuo/Documents/data/stack-out_demo/p2/ves_seg/p2_vesicle.json', excel_path='/home/liushuo/Documents/data/stack-out_demo/p2/ves_seg/p2_vesicle.xlsx')
    fire.Fire(json_to_excel)