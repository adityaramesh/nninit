local F      = require('F')
local image  = require('image')
local nninit = require('nninit')

local img = image.lena()
local img_width = img:size(2)
image.save('output/original.png', img)

for _, scale in pairs({1, 2, 3, 10, 11}) do
	local m = nninit.make_spatial_upsampling_conv({n_maps_in = 3, expansion_factor = 1,
		scale = scale, kernel = nninit.lanczos(3)})

	local new_width = math.floor(img_width / scale)
	local x = image.scale(img, new_width, new_width)

	local y = image.scale(x, scale * new_width, scale * new_width)
	image.save(F'output/scaled_output_{scale}.png', y)

	local y_hat = m:forward(x)
	image.save(F'output/conv_output_{scale}.png', y_hat)

	print(torch.norm(torch.add(y, -1 , y_hat)))
end

for _, scale in pairs({1, 2, 3, 10, 11}) do
	local m = nninit.make_spatial_upsampling_conv({kernel_width = 8 * scale,
		n_maps_in = 3, expansion_factor = 1, scale = scale, kernel = nninit.lanczos(3)})

	local new_width = math.floor(img_width / scale)
	local x = image.scale(img, new_width, new_width)

	local y = image.scale(x, scale * new_width, scale * new_width)
	image.save(F'output/scaled_output_{scale}_extra.png', y)

	local y_hat = m:forward(x)
	image.save(F'output/conv_output_{scale}_extra.png', y_hat)

	print(torch.norm(torch.add(y, -1 , y_hat)))
end
