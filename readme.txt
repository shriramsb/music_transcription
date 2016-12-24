Details of programs:
	1. frame_picker
		To get a frame from a video
		Input: video.mp4 dest_img.jpg 
		Output: dest_img.jpg
	
	2. keyboard_edge
		To manually select end points of keyboard in an image of piano
		Input: image.jpg dest.xml
		Output: dest.xml
		dest.xml stores the end points of keyboard

	3. perspective_transform
		To get the transform matrix to extract keyboard from the image
		img.jpg - image of keyboard
		cropped.jpg - extracted keyboard image
		end_pts.xml - output of keyboard_edge
		transform.xml - stores the transform matrix
		Input: img.jpg end_pts.xml transform.xml cropped.jpg
		Output: transform.xml, cropped.jpg
	
	4. key_detection
		To detect white and black keys in transformed image of keyboard
		cropped.jpg - transformed image of keyboard
		dest.xml - output xml file to store white_keys, black_keys
		Input: cropped.jpg dest.xml
		Output: dest.xml

	5. key_press_detector
		To detect the pressed keys in a video
		video.mp4 - video of piano being played
		cropped.jpg - transformed image of keyboard background
		transform.xml - xml storing transform matrix
		keys.xml - xml storing keys (output from key_detection)
		Input: video.mp4 cropped.jpg transform.xml keys.xml
	
	6. find_pressed_keys
		Same as key_press_detector
		More structured code, uses class KeyPressDetector