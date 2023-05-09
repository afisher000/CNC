Steps to creating gcode for ski-hill:

1) Choose ne, sw coordinates to use as boundary. Add to GPS coordinates.csv. Keep in mind you will want space to engrave the mountain name or logo.
2) Run generate_satellite_map.py to use the Google Maps API to create a png map of the ski mountain.
3) 	a) Open inkscape and scale artboard to model dimensions. Overlay satellite google map and match to artboard as best as possible. Can't be perfectly exact due to pixel rounding in google map vs elevation data.
	b) Set stroke width to 2mm and draw runs such that lines don't overlap. This ensures enough spacing in the final cuts. 
4) To add svg engravings:
	a) Download desired png. Crop so borders are small.
	b) Convert to svg using inkscape. 
	c) Load svg into fusion and generate engraving tool paths in a .nc file. Make the model just as large as the png. 
	d) Add the png image to contours.svg. In object properties, label the image so it can be identified from python.

4) 	Run generate_gcode.py to read the paths in contours.svg and create gcode files for roughing with stepdowns, roughing without stepdowns, smoothings, cutting contours, and engraving png images.
	 
