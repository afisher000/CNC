Steps to creating gcode for ski-hill:

1) Choose ne, sw coordinates to use for map in googlemaps.
2) Use generate_maps.py to save elevation data as png and the satellite google map as png.
3) 	a) Use Fusion360 and Image2Surface to make a model of the 2d surface with correct dimensions for wood piece.
		Have to specify the height.
	b) Generate the rough cutting gcode. Make note of model length and width.
4) 	a) Open inkscape and scale artboard to model dimensions. Overlay satellite google map and match to artboard
		as best as possible. Can't be perfectly exact due to pixel rounding in google map vs elevation data.
	b) Set stroke width to 2mm and draw runs such that lines don't overlap. This ensures enough spacing in the final 
		cuts. Before saving, reduce stroke to .5mm (or lower?) and feel free to bring lines of same color very close.
		It doesn't matter if blue runs overlap in final cuts. Export as contours.png with 800dpi.
5) 	Run generate_contour_gcode.py to process contours.png and detect points along contours. 
	It will generate gcode files for cutting the contours and smoothing based on an interpolation of the elevation data.
		 