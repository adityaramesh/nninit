local F      = require('F')
local image  = require('image')
local nninit = require('nninit')

local img = image.lena()
local img_width = img:size(2)
image.save('output/original.png', img)

for _, kw in pairs({1, 2, 3, 10, 11}) do
	local m = nninit.make_identity_spatial_conv({kernel_width = kw, n_maps_in = 3,
		expansion_factor = 1})
	local out = m:forward(img)
	image.save(F'output/output_{kw}.png', out)
end
