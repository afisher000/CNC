# %%
from lxml import etree
import os


def explode_colors(input_file='extensions applied.svg', destination_folder = 'Exploded Files'):  

    # Define color dictionary
    colors = {
        'black':'#000000',
        'red':'#ff0000',
        'green':'#00ff00',
        'blue':'#0000ff',
        'purple':'#ff00ff',
        'text':'',
    }

    # Function to check stroke color
    def is_incorrect_color(path, hexcolor):
        styles = path.attrib['style']
        for style in styles.split(';'):
            key, value = style.split(':')
            if key=='stroke':
                stroke = value
                if stroke==hexcolor:
                    return False
                else:
                    return True
                
        # Text does not have stroke in style
        if hexcolor=='':
            return False
        else:
            return True
    
    # Loop over colors, saving file for each
    for color, hexcolor in colors.items():
        # Find all paths
        tree = etree.parse(input_file)
        root = tree.getroot()
        paths = root.findall(".//{http://www.w3.org/2000/svg}path")

        # For given color, find all paths to remove
        paths_to_remove = [path for path in paths if is_incorrect_color(path, hexcolor)]

        # Remove paths
        for path in paths_to_remove:
            parent = path.getparent()
            parent.remove(path)

        # Save to file
        output_file = color+'.svg'
        file_path = os.path.join(destination_folder, output_file)
        tree.write(file_path, xml_declaration=True, encoding="UTF-8")
        print(f"Saved to {file_path}")


explode_colors(input_file='solid lines for coloring.svg', destination_folder='Coloring Exploded Files')


# %%
