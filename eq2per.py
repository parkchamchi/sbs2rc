"""
	This file is a modification of
	https://github.com/fuenwang/Equirec2Perspec/blob/master/Equirec2Perspec.py

	original license:

		MIT License

		Copyright (c) 2021 Fu-En Wang

		Permission is hereby granted, free of charge, to any person obtaining a copy
		of this software and associated documentation files (the "Software"), to deal
		in the Software without restriction, including without limitation the rights
		to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
		copies of the Software, and to permit persons to whom the Software is
		furnished to do so, subject to the following conditions:

		The above copyright notice and this permission notice shall be included in all
		copies or substantial portions of the Software.

		THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
		IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
		FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
		AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
		LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
		OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
		SOFTWARE.

"""

import cv2
import numpy as np

def xyz2lonlat(xyz):
	atan2 = np.arctan2
	asin = np.arcsin

	norm = np.linalg.norm(xyz, axis=-1, keepdims=True)
	xyz_norm = xyz / norm
	x = xyz_norm[..., 0:1]
	y = xyz_norm[..., 1:2]
	z = xyz_norm[..., 2:]

	lon = atan2(x, z)
	lat = asin(y)
	lst = [lon, lat]

	out = np.concatenate(lst, axis=-1)
	return out

def lonlat2XY(lonlat, shape):
	X = (lonlat[..., 0:1] / (2 * np.pi) + 0.5) * (shape[1] - 1)
	Y = (lonlat[..., 1:] / (np.pi) + 0.5) * (shape[0] - 1)
	lst = [X, Y]
	out = np.concatenate(lst, axis=-1)

	return out 

class Equirect():
	def __init__(self, origh, origw, outsize, fov=90, theta=0, phi=0, interpolation=cv2.INTER_LINEAR):
		self.origw = origw
		self.origh = origh
		self.outsize = outsize
		self.fov = fov
		self.theta = theta
		self.phi = phi
		self.interpolation = interpolation

		f = 0.5 * outsize * 1 / np.tan(0.5 * fov / 180.0 * np.pi)
		cx = cy = (outsize - 1) / 2.0
		K = np.array([
				[f, 0, cx],
				[0, f, cy],
				[0, 0,  1],
			], np.float32)
		K_inv = np.linalg.inv(K)
		
		x = np.arange(outsize)
		y = np.arange(outsize)
		x, y = np.meshgrid(x, y)
		z = np.ones_like(x)
		xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1)
		xyz = np.dot(xyz, K_inv.T)

		y_axis = np.array([0.0, 1.0, 0.0], np.float32)
		x_axis = np.array([1.0, 0.0, 0.0], np.float32)
		R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
		R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
		R = np.dot(R2, R1)
		xyz = np.dot(xyz, R.T)
		lonlat = xyz2lonlat(xyz) 
		XY = lonlat2XY(lonlat, shape=(origh, origw)).astype(np.float32)

		self.XY = XY

	def transform(self, img):
		if img.shape[:2] != (self.origh, self.origw):
			raise ValueError("Received {}, expecting {}.".format(img.shape[:2], (self.origh, self.origw)))

		return cv2.remap(img, self.XY[..., 0], self.XY[..., 1], self.interpolation, borderMode=cv2.BORDER_WRAP)