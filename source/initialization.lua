--[[
Adapted from `https://github.com/Kaixhin`.
--]]

local F = require('F')
local nn = require('nn')

--[[
Returns the fan-in and the fan-out of the given module. The fan-in is the number of inputs used to
compute one activation; the fan-out is the number of activations used to compute one output.
--]]
local function compute_fan(module)
	local typename = torch.typename(module)

	if typename == 'nn.Linear'       or
	   typename == 'nn.LinearNoBias' or
	   typename == 'nn.LookupTable'  then
		return module.weight:size(2), module.weight:size(1)
	elseif typename == 'nn.Bilinear' then
		return module.weight:size(2) * module.weight:size(3), module.weight:size(1)
	elseif typename:find('TemporalConvolution') then
		return module.weight:size(2), module.weight:size(1)
	elseif typename:find('SpatialConvolution')        or
	       typename:find('SpatialFullConvolution')    or
	       typename:find('SpatialDilatedConvolution') then
		local kernel_size = module.kW * module.kH
		return module.nInputPlane * kernel_size, module.nOutputPlane * kernel_size
	elseif typename:find('VolumetricConvolution') then
		local kernel_size = module.kT * module.kW * module.kH
		return module.nInputPlane * kernel_size, module.nOutputPlane * kernel_size
	else
		error(F"Unsupported module type '{typename}'.")
	end
end

local function init_tensor(tensor, dist, variance)
	if dist == 'normal' then
		tensor:normal(0, math.sqrt(variance))
	elseif dist == 'uniform' then
		local a = math.sqrt(3 * variance)
		tensor:uniform(-a, a)
	else
		error(F"Unsupported distribution '{dist}'.")
	end
end

nn.Module.init = function(self, initializer, ...)
	initializer(self, ...)
	return self
end

--[[
Let $f$ be a given activation function, and $x$ be a random variable with zero mean and unit
variance. For simplicity, assume that $x$ is never in the saturating domain of $f$. This function
returns $1 / \Var{f(x)}$.
--]]
function nninit.compute_gain(act_func, args)
	if act_func == 'identity' or act_func == 'tanh' then
		return 1
	elseif act_func == 'relu' then
		return 2
	elseif act_func == 'lrelu' or act_func == 'prelu' then
		return 2 / (1 + args.leak^2)
	elseif act_func == 'rrelu' then
		local avg_leak = (args.leak_min + args.leak_max) / 2
		return 2 / (1 + avg_leak^2)
	else
		error(F"Unsupported activation function '{act_func}'")
	end
end

--[[
Generalizes several popular initialization schemes:

    Scheme               Gain        Variant
    [LeCun et al.][1]    1           1
    [Glorot et al.][2]   1           3
    [He et al.][3]       variable    1 or 2

Important notes:
* The input is not usually passed through an activation function, so the value of the gain for the
  first layer should typically be one, which is the value returned by `compute_gain('identity')`.

[1]: http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf
"Efficient BackProp"

[2]: http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf
"Understanding the Difficulty of Training Deep Feedforward Neural Networks"

[3]: https://arxiv.org/abs/1502.01852
"Delving Deep Into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification"
--]]
function nninit.preserve_variance(module, gain, variant, dist)
	variant = variant or 1
	dist = dist or 'normal'
	local fan_in, fan_out = compute_fan(module)

	if variant == 1 then
		init_tensor(module.weight, dist, gain / fan_in)
	elseif variant == 2 then
		init_tensor(module.weight, dist, gain / fan_out)
	elseif variant == 3 then
		init_tensor(module.weight, dist, 2 * gain / (fan_in + fan_out))
	else
		error(F"Unsupported variant '{variant}'.")
	end

	if module.bias ~= nil then
		module.bias:zero()
	end
end

--[[
Simplified version of the initialization scheme described in [Saxe et al.][1]. We do not share
orthogonal matrices across layers as described in the paper or by Andrew Saxe in [this Google+
discussion][2].

It is not clear how to extend orthogonal initialization as described in the paper to convolutional
layers. One way to go about this would be to view orthogonal initiazation for a linear layer as
enforcing that the "feature extractors" (i.e., the rows of the weight matrix) are orthonormal. To
apply this logic to convolutional layers, we can view the weights of each convolutional kernel as a
vector, and ensure that all of these vectors are orthonormal.

[1]: http://arxiv.org/abs/1312.6120
"Exact Solutions to the Nonlinear Dynamics of Learning in Deep Linear Neural Networks"

[2]: https://plus.google.com/+SoumithChintala/posts/RZfdrRQWL6u
--]]
function nninit.orthogonal(module, gain)
	assert(target:nDimension() >= 2)

	local w = module.weight
	local rows = w:size(1)
	local cols = w:size(2)
	for i = 3, w:nDimension() do cols = cols * w:size(i) end

	local seed = w.new(rows, cols):normal(0, 1)
	local u, _, v = torch.svd(seed, 'S')

	local q
	if q:size(2) == cols then q = u
	else q = v end

	assert(q:size(1) == rows)
	assert(q:size(2) == cols)

	q:resizeAs(w):mul(math.sqrt(gain))
	w:copy(q)

	if module.bias ~= nil then
		module.bias:zero()
	end
end

local function validate_common_conv_args(args)
	local kw    = args.kernel_width
	local fm_in = args.n_maps_in
	local k     = args.expansion_factor

	assert(kw >= 1)
	assert(fm_in >= 1)
	assert(k >= 1, "Only positive, integral expansion factors are supported.")
	return kw, fm_in, k
end

--[[
Returns an `nn.SpatialConvolution` layer initialized such that each group of $k$ consecutive output
feature maps copies the same input feature map.

TODO: Support for perforated convolutions?
--]]
function nninit.make_identity_spatial_conv(args)
	local kw, fm_in, k = validate_common_conv_args(args)
	local fm_out = k * fm_in
	local dw = 1

	--[[
	The padding is chosen such that the application of the kernel to the top-left corner of the
	padded input image results in the top-left pixel of the non-padded image coinciding with the
	single nonzero entry of the kernel, and similarly for the bottom-right corner of the image.
	The case where the kernel width is even forces us to make an arbitrary choice about where to
	place the nonzero entry.
	--]]

	if kw % 2 == 1 then
		local pw = (kw - 1) / 2
		local m = nn.SpatialConvolution(fm_in, fm_out, kw, kw, dw, dw, pw, pw)
		local w, b = m.weight

		w:zero()
		b:zero()

		for i = 1, fm_in do
			for j = 1, k do
				local i_o = k * (i - 1) + j
				w[i_o][i][(kw + 1) / 2][(kw + 1) / 2] = 1
			end
		end

		return m
	else
		local pw = 0
		local m = nn.SpatialConvolution(fm_in, fm_out, kw, kw, dw, dw, pw, pw)
		local w, b = m.weight

		w:zero()
		b:zero()

		for i = 1, fm_in do
			for j = 1, k do
				local i_o = k * (i - 1) + j
				w[i_o][i][(kw - 2) / 2][(kw - 2) / 2] = 1
			end
		end

		local p_lt = (kw - 2) / 2
		local p_rb = (kw + 2) / 2
		return nn.Sequential():add(nn.SpatialZeroPadding(p_lt, p_rb, p_lt, p_rb)):add(m)
	end
end

--[[
Returns a strided `nn.SpatialConvolution` layer initialized such that each group of $k$ consecutive
output feature maps downsamples the same input feature map.

TODO: Support for perforated convolutions?
--]]
function nninit.make_downsampling_spatial_conv(args)
	local kw, fm_in, k = validate_common_conv_args(args)
	local fm_out = k * fm_in
	local scale = args.scale

	assert(scale >= 1)
	assert(kw >= scale)
	local extra_width = kw - scale

	if extra_width % 2 == 0 then
		local pw = extra_width / 2
		local m = nn.SpatialConvolution(fm_in, fm_out, kw, kw, scale, scale, pw, pw)
		local w, b = m.weight, m.bias()

		w:zero()
		b:zero()

		for i = 1, fm_in do
			for j = 1, k do
				local i_o = k * (i - 1) + j
				w[{{i_o}, {i}, {pw + 1, pw + kw}, {pw + 1, pw + kw}}]:fill(1 / kw^2)
			end
		end

		return m
	else
		local pw = 0
		local m = nn.SpatialConvolution(fm_in, fm_out, kw, kw, scale, scale, pw, pw)
		local w, b = m.weight

		w:zero()
		b:zero()

		local p_lt = (kw - 2) / 2
		local p_rb = (kw + 2) / 2

		for i = 1, fm_in do
			for j = 1, k do
				local i_o = k * (i - 1) + j
				w[{{i_o}, {i}, {p_lt + 1, p_lt + kw}, {p_rb + 1, p_rb + kw}}]:fill(1 / kw^2)
			end
		end

		return nn.Sequential():add(nn.SpatialZeroPadding(p_lt, p_rb, p_lt, p_rb)):add(m)
	end
end

--[[
Returns a `nn.SpatialFullConvolution` layer initialized such that each group of $k$ consecutive
output feature maps upsamples the same input feature map.
--]]
function nninit.make_spatial_upsampling_conv(args)
	local kw, fm_in, k = validate_common_conv_args(args)
	local fm_out = k * fm_in
	local scale = args.scale

	assert(scale >= 1)
	local inner_width = 2 * scale - 1
	assert(kw >= inner_width)
	local extra_width = kw - inner_width

	--[[
	If `kw > inner_width`, then the part of the kernel outside the inner `inner_width x
	inner_width` block is set to zero. But we don't want these zeros to appear in the border of
	the output image, so we define some additional padding parameters in order to remove them.
	--]]
	local p_lt_extra, p_rb_extra

	if extra_width % 2 == 0 then
		p_lt_extra, p_rb_extra = extra_width / 2
	else
		p_lt_extra = (extra_width - 1) / 2
		p_rb_extra = (extra_width + 1) / 2
	end

	local pw, m

	if scale % 2 == 0 then
		pw = scale + scale / 2
		m = nn.SpatialFullConvolution(fm_in, fm_out, kw, kw, scale, scale, pw, pw, 1, 1)
	else
		pw = scale + (scale - 1) / 2
		m = nn.SpatialFullConvolution(fm_in, fm_out, kw, kw, scale, scale, pw, pw, 0, 0)
	end

	local w, b = m.weight, m.bias
	w:zero()
	b:zero()

	local v = torch.Tensor(inner_width)
	for i = 1, kw do v[i] = 1 - math.abs(i - scale) / scale end
	local init = torch.ger(v, v)

	for i = 1, fm_in do
		for j = 1, k do
			w[{{i_o}, {i}, {pw + 1, pw + inner_width}, {pw + 1, pw + inner_width}}]:copy(init)
		end
	end

	if extra_width % 2 == 0 then
		local pw_extra = 1 + extra_width / 2
		return nn.Sequential():add(nn.SpatialReplicationPadding(
			pw_extra, pw_extra, pw_extra, pw_extra)):add(m)
	else
		local p_lt_extra = 1 + (extra_width - 1) / 2
		local p_rb_extra = 1 + (extra_width - 1) / 2
		return nn.Sequential():add(nn.SpatialReplicationPadding(
			p_lt_extra, p_rb_extra, p_lt_extra, p_rb_extra)):add(m)
	end
end

-- TODO how to get this method to reduce to LSUV?
function nninit.data_driven(module, update_func, dampening)
	dampening = dampening or 0.25
end

return nninit
