Steps to creating gcode for ski-hill:

1) Choose ne, sw coordinates to use for map in googlemaps. Add to GPS coordinates.csv.
2) Run generate_satellite_map.py to use the Google Maps API to create a png map of the ski mountain.
3) 	a) Open inkscape and scale artboard to model dimensions. Overlay satellite google map and match to artboard as best as possible. Can't be perfectly exact due to pixel rounding in google map vs elevation data.
	b) Set stroke width to 2mm and draw runs such that lines don't overlap. This ensures enough spacing in the final cuts. Before saving, reduce stroke and feel free to bring lines of same color very close. Export as contours.png with 800dpi.
4) 	Run generate_gcode.py to read the paths in contours.svg and create gcode files for roughing with stepdowns, roughing without stepdowns, smoothings, and cutting contours.		 
