import numpy as np
import cv2

import eq2per as eq

import time
from datetime import datetime

class SBS2RC():
	"""
	Convert a side-by-side 3D image to a red-cyan (analglyph) image.
	"""

	def __init__(self, vertical=False, method="ch", proj=None, switch=False):
		"""
		:param vertical: assumes a vertically stacked image.
		:param method:
			"ch": discard color channels that do not match.
			"gr": set the image to grayscale then apply to the matching channels (left: red, right: cyan)			
		:param proj: a Projection object.
		:param switch: Switch eyes.
		"""

		self.vertical = vertical
		self.method = method
		self.proj = proj
		self.switch = switch

	def transform(self, img):
		height, width = img.shape[:2] #size of the original input

		if not self.vertical:
			newheight, newwidth = height, width//2 #newheight, newwidth: size per eye 

			left = img[:, :newwidth]
			right = img[:, newwidth:]
		else:
			newheight, newwidth = height//2, width

			left = img[:newheight, :]
			right = img[newheight:, :]

		sides = []
		masks = [(0, 0, 1), (1, 1, 0)] #B, G, R
		if self.switch:
			masks = masks[::-1]
		for side, mask in zip([left, right], masks):
			if self.method == "ch":
				for i, val in enumerate(mask):
					if not val:
						side[:, :, i] = 0

			elif self.method == "gr":
				gray = cv2.cvtColor(side, cv2.COLOR_BGR2GRAY)
				side = np.zeros((newheight, newwidth, 3), dtype=np.uint8)
				for i, val in enumerate(mask):
					if val:
						side[:, :, i] = gray

			#project
			if self.proj:
				side = self.proj.transform(side)
			
			sides.append(side)

		out = sides[0] + sides[1]

		return out

class Projection():
	"""
	Convert 180- or 360- video with equirectangular projection into a flat screen.
	"""

	def __init__(self, is180):
		self.is180 = is180
		self.origw = self.origh = 0

	def pad(self, img): #for 180
		height, width = img.shape[:2]

		out = np.zeros((height, width*2, 3), dtype=np.uint8)
		out[:, width//2:width//2+width] = img

		return out

	def transform(self, img):
		#check if the image size is changed
		if img.shape[:2] != (self.origh, self.origw if not self.is180 else self.origw//2):
			self.origh, self.origw = img.shape[:2]
			if self.is180:
				self.origw *= 2 #pad

			self.getEqs()

		if self.is180:
			img = self.pad(img)	

		img = self.method(img)
		return img

class CubemapProjection(Projection):

	def getEqs(self):
		"""
		Get Equirect objects. Should be called every time the image shape is changed.
		"""

		self.size = self.origh//2 #size of the output.

		params = { # (theta, phi)
			"front": (0, 0),
			"left": (-90, 0), 
			"right": (90, 0), 
			"back": (180, 0),
			"up":  (0, 90),
			"down": (0, -90),
		}

		for name, (theta, phi) in params.items():
			params[name] = eq.Equirect(self.origh, self.origw, self.size, 90, theta, phi) #90 fov approximates a half of the height.

		self.eqs = params

	def method(self, img):
		size = self.size

		cubes = {}
		for direction, equ in self.eqs.items():
			if direction == "back" and self.is180:
				continue

			cubes[direction] = equ.transform(img)

		out = np.zeros((size*3, size*4, 3), dtype=np.uint8)
		locs = {
			"front": (1, 1),
			"left": (0, 1), 
			"right": (2, 1), 
			"back": (3, 1),
			"up": (1, 0),
			"down": (1, 2)
		}

		for direction, (x, y) in locs.items():
			if self.is180 and direction == "back":
				continue

			x = int(x*size)
			y = int(y*size)

			out[y:y+size, x:x+size] = cubes[direction]

		#roll so that the front is moved to the center
		out = np.roll(out, size//2, axis=1)

		if self.is180:
			#unpad
			height, width = out.shape[:2]
			out = out[int(height*(1/6)):int(height*(5/6)), int(width*(1/4)):int(width*(3/4))]

		return out	

class FlatProjection(Projection):
	def __init__(self, is180, fov=90, theta=0, phi=0):
		super().__init__(is180)

		self.fov = fov
		self.theta = theta
		self.phi = phi

	def getEqs(self):
		self.size = self.origh
		self.eq = eq.Equirect(self.origh, self.origw, self.origh, self.fov, self.theta, self.phi)

	def method(self, img):
		out = self.eq.transform(img)
		return out

def makeImage(inp, outp, transformer, preview=False, scale=None):
	img = cv2.imread(inp)
	#scale
	if scale:
		height, width = img.shape[:2]
		width = int(width*scale)
		height = int(height*scale)
		img = cv2.resize(img, (width, height))

	img = transformer.transform(img)

	if not preview:
		cv2.imwrite(outp, img)
	else:
		print("Press any key to end.")
		cv2.imshow("preview", img)
		cv2.waitKey(0)

def makeVideo(inp, outp, transformer, fourcc="XVID", preview=False, scale=None):
	"""
	:param input: filename
	:param output: filename
	:param transformer: transformer to process the frame.
	:param fourcc: the fourcc code. Defaults to "XVID".
	:param preview: show ouput without saving to disk.
	:parma scale: scale the input. (floating point)
	"""

	print(datetime.now())
	if preview:
		print("Press 'q' to end.")
	
	cap = cv2.VideoCapture(inp)

	vout = None
	framenum = 1
	start = time.time()
	while cap.isOpened():
		ret, frame = cap.read()

		if not ret:
			print("\nCan't receive frame (stream end?)")
			break

		if not vout: #init
			vout = True
			width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
			height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
			fps = cap.get(cv2.CAP_PROP_FPS)
			#def_fourcc = cap.get(cv2.CAP_PROP_FOURCC)
			framecount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

			if scale:
				width = int(width*scale)
				height = int(height*scale)

			#Pass the transformer a dummy image to get the output size.
			outheight, outwidth = transformer.transform(np.empty((height, width, 3), dtype=np.uint8)).shape[:2]
			size = (outwidth, outheight)

			if not preview:
				fourcc = cv2.VideoWriter_fourcc(*fourcc)
				vout = cv2.VideoWriter(outp, fourcc, fps, size)

		#scale
		if scale:
			frame = cv2.resize(frame, (width, height))

		out = transformer.transform(frame)

		if out.shape[:2] != size[::-1]:
			print("\nERROR: Received {}, expecting {}.".format(out.shape[:2], size[::-1]))
			break
		
		if not preview:
			vout.write(out)
		else:
			cv2.imshow("preview", out)
			if cv2.waitKey(1) == ord('q'):
				break

		print("\r{}%... ({}/{})".format(int(framenum/framecount * 100), framenum, framecount), end='', flush=True)
		framenum += 1

	cap.release()
	if vout and not preview:
		vout.release()
	cv2.destroyAllWindows()

	print("\nTook {} secs.".format(int(time.time() - start)))

#####################################################################

if __name__ == "__main__":
	import argparse
	import os

	parser = argparse.ArgumentParser(description="Converts a side-by-side 3D video to Analglyph (red-cyan) video.")

	parser.add_argument("inputname", help="The filename of the input")
	parser.add_argument("-o", "--out", help="The filename of the output")
	parser.add_argument("-i", "--image", help="Assume an image input", action="store_true")
	
	parser.add_argument("-g", "--guess", help="Guess parameters from the filename. This will suppress -v and -p.", action="store_true")
	parser.add_argument("-v", "--vertical", help="Assume a vertically stacked format", action="store_true")
	parser.add_argument("-p", "--project", help="Project a 180 or 360 degree format (about twice slower). Assumes Equirectangular projection.", type=int, choices=[180, 360])

	parser.add_argument("--method", help="""
		Coloring method. default: %(default)s
		<"ch": discard color channels that do not match.>,
		<"gr": set the image to grayscale then apply to the matching channels (left: red, right: cyan)>
		""", choices=["ch", "gr"], default="ch")

	parser.add_argument("--projmethod", help="Projection method. default: %(default)s", choices=["cubemap", "flat"], default="cubemap")
	parser.add_argument("--fov", help="FOV value used for --projmethod flat. default: %(default)s", type=int, default=90)
	parser.add_argument("--theta", help="theta value used for --projmethod flat. default: %(default)s", type=int, default=0)
	parser.add_argument("--phi", help="phi value used for --projmethod flat. default: %(default)s", type=int, default=0)

	parser.add_argument("--scale", help="Scale the input (%%). default: %(default)s", type=int, default=100)
	parser.add_argument("--switch", help="Switch eyes.", action="store_true")
	parser.add_argument("--fourcc", help="The fourcc code of the output file. default: %(default)s (if it doesn't work, try \"XVID\" with .avi)", default="mp4v")
	parser.add_argument("--preview", help="Show the output without saving", action="store_true")
	parser.add_argument("--noaudio", help="Do not add the audio.", action="store_true")
	
	args = parser.parse_args()

	#check the inputname
	inputname = args.inputname
	if not os.path.exists(inputname) or not os.path.isfile(inputname):
		raise ValueError("{} does not exist or is not a file.".format(inputname))

	#check the out
	
	outputname = args.out
	if not outputname:
		root, ext = os.path.splitext(inputname)
		outputname = "{}_{}{}".format(root, "sbs2rc", ext)

	if os.path.exists(outputname):
		i = 1
		root, ext = os.path.splitext(outputname)
		while True:
			newoutputname = "{}({}){}".format(root, i, ext)
			if not os.path.exists(newoutputname):
				outputname = newoutputname
				break
			i += 1

	if not args.preview:
		print("Output: {}".format(outputname))

	vertical = project = is180 = False

	#guess from the filename
	if args.guess:
		basename = os.path.basename(inputname).lower()

		keywords_dict = {
			"vertical": ["tb", "top + bottom", "3dv", "_ou.", "_ou_", "-ou.", "-ou_", "-ou-"],
			"180": ["180"],
			"360": ["360"]
		}

		for var, keywords in keywords_dict.items():
			if any([keyword in basename for keyword in keywords]):
				if var == "vertical":
					vertical = True
				else:
					project = True
					if var == "180":
						is180 = True
	else:
		#check the vertical
		vertical = args.vertical

		#check the projection
		if args.project:
			project = True
			is180 = (args.project == 180)

	if vertical:
		print("Vertical input.")
	if project: 
		if is180:
			print("Projecting 180.")
		else:
			print("Projecting 360.")

	if project:
		projections = {
			"cubemap": CubemapProjection(is180=is180),
			"flat": FlatProjection(is180=is180, fov=args.fov, theta=args.theta, phi=args.phi),
		}
		proj = projections[args.projmethod]
	else:
		proj = None

	#check scale
	if args.scale == 100:
		scale = None
	elif args.scale <= 0 or args.scale > 100:
		raise ValueError("Invalid scale value: {}.".format(args.scale))
	else:
		print("Scaling {}%.".format(args.scale))
		scale = args.scale / 100

	transformer = SBS2RC(vertical=vertical, method=args.method, proj=proj, switch=args.switch)

	###############################################
	# Process
	###############################################

	#image file
	if args.image:	
		makeImage(inputname, outputname, transformer, preview=args.preview, scale=scale)
	else:
		makeVideo(inputname, outputname, transformer, fourcc=args.fourcc, preview=args.preview, scale=scale)

	if args.preview:
		exit(1)

	print("Wrote {}".format(outputname))

	if not args.image and not args.noaudio:
		print("Adding audio...")

		try:
			import uuid

			tmpname = "tmp_sbs2rc_" + uuid.uuid4().hex + os.path.splitext(outputname)[1]
			outroot = os.path.split(outputname)[0]
			tmpname = os.path.join(outroot, tmpname) #make tmpname to be in the same folder with output
			os.system("ffmpeg -i {} -i {} -c copy -map 1:v:0 -map 0:a:0 -shortest {} -hide_banner -loglevel error".format(inputname, outputname, tmpname))
			os.replace(tmpname, outputname)

		except Exception as exc:
			print("Failed to add the audio.")
			print(exc)

		else:
			print("Successfully added the audio.")

		finally:
			#cleanup
			if os.path.exists(tmpname):
				os.remove(tmpname)