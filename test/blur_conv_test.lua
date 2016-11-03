local F      = require('F')
local image  = require('image')
local nninit = require('nninit')

local img = image.lena()
local img_width = img:size(2)
image.save('output/original.png', img)

for _, size in pairs({8, 16, 32, 64, 128, 256}) do
	local m = nninit.make_spatial_blur_conv({n_maps_in = 3, std = 0.50, input_width = size})
	local x = image.scale(img, size, size)
	image.save(F'output/output_{size}.png', x)
	local y = m:forward(x):view(-1, size, size)
	image.save(F'output/blur_output_{size}.png', y)
end
