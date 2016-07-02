local F      = require('F')
local image  = require('image')
local nninit = require('nninit')

local img = image.lena()
local img_width = img:size(2)
image.save('original.png', img)

for _, scale in pairs({1, 2, 3, 10, 11}) do
	local m = nninit.make_downsampling_spatial_conv({kernel_width = scale,
		n_maps_in = 3, expansion_factor = 1, scale = scale, input_width = img_width})

	local new_width = math.ceil(img_width / scale)
	local y = image.scale(img, new_width, new_width)
	image.save(F'output/scaled_output_{scale}.png', y)

	local y_hat = m:forward(img)
	image.save(F'output/conv_output_{scale}.png', y_hat)

	local rescaled_y = image.scale(y, 512, 512)
	local rescaled_y_hat = image.scale(y_hat, 512, 512)
	image.save(F'output/rescaled_scaled_output_{scale}.png', rescaled_y)
	image.save(F'output/rescaled_conv_output_{scale}.png', rescaled_y_hat)

	print(torch.norm(torch.add(y, -1 , y_hat)))
end

for _, scale in pairs({1, 2, 3, 10, 11}) do
	local m = nninit.make_downsampling_spatial_conv({kernel_width = 2 * scale,
		n_maps_in = 3, expansion_factor = 1, scale = scale, input_width = img_width})

	local new_width = math.ceil(img_width / scale)
	local y = image.scale(img, new_width, new_width)
	image.save(F'output/scaled_output_{scale}_padded_even.png', y)

	local y_hat = m:forward(img)
	image.save(F'output/conv_output_{scale}_padded_even.png', y_hat)

	local rescaled_y = image.scale(y, 512, 512)
	local rescaled_y_hat = image.scale(y_hat, 512, 512)
	image.save(F'output/rescaled_scaled_output_{scale}_padded_even.png', rescaled_y)
	image.save(F'output/rescaled_conv_output_{scale}_padded_even.png', rescaled_y_hat)

	print(torch.norm(torch.add(y, -1 , y_hat)))
end

for _, scale in pairs({1, 2, 3, 10, 11}) do
	local m = nninit.make_downsampling_spatial_conv({kernel_width = 2 * scale + 1,
		n_maps_in = 3, expansion_factor = 1, scale = scale, input_width = img_width})

	local new_width = math.ceil(img_width / scale)
	local y = image.scale(img, new_width, new_width)
	image.save(F'output/scaled_output_{scale}_padded_odd.png', y)

	local y_hat = m:forward(img)
	image.save(F'output/conv_output_{scale}_padded_odd.png', y_hat)

	local rescaled_y = image.scale(y, 512, 512)
	local rescaled_y_hat = image.scale(y_hat, 512, 512)
	image.save(F'output/rescaled_scaled_output_{scale}_padded_odd.png', rescaled_y)
	image.save(F'output/rescaled_conv_output_{scale}_padded_odd.png', rescaled_y_hat)

	print(torch.norm(torch.add(y, -1 , y_hat)))
end
