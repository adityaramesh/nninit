--[[
Adapted from `https://github.com/Kaixhin`.
--]]

local F     = require('F')
local nn    = require('nn')
local xlua  = require('xlua')
local image = require('image')

local lib = cudnn or nn
local SpatialConvolution = lib.SpatialConvolution
local SpatialFullConvolution = lib.SpatialFullConvolution

--[[
Returns the fan-in and the fan-out of the given module. The fan-in is the number of inputs used to
compute one activation; the fan-out is the number of activations computed using one output.
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

local nninit = {}

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
function nninit.normalize_variance(module, gain, variant, dist)
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
	local w = module.weight
	local rows, cols

	if not torch.typename(module):find('SpatialFullConvolution') then
		rows, cols = w:size(1), w:size(2)
		for i = 3, w:nDimension() do cols = cols * w:size(i) end
	else
		rows, cols = w:size(2), w:size(1)
		for i = 3, w:nDimension() do cols = cols * w:size(i) end
	end

	local seed = w.new(rows, cols):normal(0, 1)
	local u, _, v = torch.svd(seed, 'S')
	local q

	if u:size(2) == cols then
		q = u
	else
		q = v:t()

		--[[
		XXX: As of the time of writing, if Torch is set up to use MKL for BLAS, then using
		the 'S' option with SVD can yield the same results as 'A', even when this should not
		be the case. For example, try SVD with the 'S' option on a 200 x 4096 matrix -- U is
		4096 x 4096, when it should be 4096 x 200.

		Remove this workaround after the bug mentioned above is fixed.
		--]]
		if rows ~= cols and q:size(1) == q:size(2) then
			q = q[{{1, math.min(rows, cols)}, {}}]
		end
	end

	assert(
		(q:size(1) == rows and q:size(2) == math.min(rows, cols)) or
		(q:size(1) == math.min(rows, cols) and q:size(2) == cols)
	)

	q:resizeAs(w):mul(math.sqrt(gain))
	if torch.typename(module):find('SpatialFullConvolution') then q = q:transpose(1, 2):clone() end
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
Returns an `SpatialConvolution` layer initialized such that each group of $k$ consecutive output
feature maps copies the same input feature map.

TODO: Support for non-square inputs.
TODO: Support for perforated convolutions?
--]]
function nninit.make_identity_spatial_conv(args)
	local kw, fm_in, k = validate_common_conv_args(args)
	local fm_out       = k * fm_in
	local dw           = 1

	--[[
	The padding is chosen such that the application of the kernel to the top-left corner of the
	padded input image results in the top-left pixel of the non-padded image coinciding with the
	single nonzero entry of the kernel, and similarly for the bottom-right corner of the image.
	The case where the kernel width is even forces us to make an arbitrary choice about where to
	place the nonzero entry.
	--]]

	if kw % 2 == 0 then
		local pw = 0
		local m = SpatialConvolution(fm_in, fm_out, kw, kw, dw, dw, pw, pw)
		local w, b = m.weight, m.bias

		w:zero()
		b:zero()

		for i = 1, fm_in do
			for j = 1, k do
				local i_out = k * (i - 1) + j
				w[i_out][i][kw / 2][kw / 2] = 1
			end
		end

		local p_lt = kw / 2 - 1
		local p_rb = kw / 2

		return nn.Sequential():
			add(nn.SpatialZeroPadding(p_lt, p_rb, p_lt, p_rb)):
			add(m)
	else
		local pw = (kw - 1) / 2
		local m = SpatialConvolution(fm_in, fm_out, kw, kw, dw, dw, pw, pw)
		local w, b = m.weight, m.bias

		w:zero()
		b:zero()

		for i = 1, fm_in do
			for j = 1, k do
				local i_out = k * (i - 1) + j
				w[i_out][i][(kw + 1) / 2][(kw + 1) / 2] = 1
			end
		end

		return m
	end
end

--[[
Returns a strided `SpatialConvolution` layer initialized such that each group of $k$ consecutive
output feature maps downsamples the same input feature map.

Note: this doesn't match the result of upsampling using `image.scale`, but (1) I don't have time to
examine the source to see what it's doing, and (2) the output from this function seems to "move" the
original image less and retains more sharpness in the examples that I tested.

TODO: Support for non-square inputs.
TODO: Support for perforated convolutions?
--]]
function nninit.make_spatial_downsampling_conv(args)
	local kw, fm_in, k = validate_common_conv_args(args)
	local fm_out       = k * fm_in
	local scale        = args.scale
	local iw           = args.input_width

	assert(scale >= 1)
	assert(kw >= scale)
	assert(iw >= kw)

	local extend_width = 0
	if iw % scale ~= 0 then extend_width = scale - iw % scale end
	local extra_width = kw - scale

	local p_lt, p_rb

	if extend_width % 2 == 0 then
		p_lt, p_rb = extend_width / 2, extend_width / 2
	else
		p_lt, p_rb = (extend_width - 1) / 2, (extend_width + 1) / 2
	end

	if extra_width % 2 == 0 then
		local pw = extra_width / 2
		local m = SpatialConvolution(fm_in, fm_out, kw, kw, scale, scale, pw, pw)
		local w, b = m.weight, m.bias

		w:zero()
		b:zero()

		for i = 1, fm_in do
			for j = 1, k do
				local i_out = k * (i - 1) + j
				w[{{i_out}, {i}, {pw + 1, pw + scale},
					{pw + 1, pw + scale}}]:fill(1 / scale^2)
			end
		end

		if extend_width == 0 then
			return m
		else
			return nn.Sequential():
				add(nn.SpatialReflectionPadding(p_lt, p_rb, p_lt, p_rb)):
				add(m)
		end
	else
		local pw = 0
		local m = SpatialConvolution(fm_in, fm_out, kw, kw, scale, scale, pw, pw)
		local w, b = m.weight, m.bias

		w:zero()
		b:zero()

		local zp_lt = (extra_width - 1) / 2
		local zp_rb = (extra_width + 1) / 2

		for i = 1, fm_in do
			for j = 1, k do
				local i_out = k * (i - 1) + j
				w[{{i_out}, {i}, {zp_lt + 1, zp_lt + scale},
					{zp_lt + 1, zp_lt + scale}}]:fill(1 / scale^2)
			end
		end

		if extend_width == 0 then
			return nn.Sequential():
				add(nn.SpatialZeroPadding(zp_lt, zp_rb, zp_lt, zp_rb)):
				add(m)
		else
			return nn.Sequential():
				add(nn.SpatialReflectionPadding(p_lt, p_rb, p_lt, p_rb)):
				add(nn.SpatialZeroPadding(zp_lt, zp_rb, zp_lt, zp_rb)):
				add(m)
		end
	end
end

--[[
Returns a `SpatialFullConvolution` layer initialized such that each group of $k$ consecutive output
feature maps upsamples the same input feature map.

TODO: Support for non-square inputs.
--]]
function nninit.make_spatial_upsampling_conv(args)
	local kw, fm_in, k = validate_common_conv_args(args)
	local fm_out       = k * fm_in
	local scale        = args.scale

	assert(scale >= 1)
	local inner_width = 2 * scale - 1
	assert(kw >= inner_width)
	local extra_width = kw - inner_width

	--[[
	Some explanations for the variables:
	  - `extra_width`: If `kw > inner_width`, then in order to produce the same output as the
	    case where `kw == inner_width`, we initialize the bottom-left `inner_width x
	    inner_width` block of the kernel to perform bilinear upsampling, and set the remaining
	    border along the top and right of the image to zero.
	  - `aw`: Note that the width of the "pure" part of the output image that is not affected by
	    the `SpatialReplicationPadding` is given by `o_w = scale * (i_w - 1) + 1`. The desired
	    width of the output image is `scale * i_w`, so we must extend the width of the image by
	    `scale - 1` "impure" pixels that are computed partially using the padding values. If
	    `scale` is even, then we can use an equal number of "impure" pixels from all sides of
	    the image, so there is no asymmetry. Otherwise, we choose arbitrarily to use an extra
	    row and column from the bottom and right side of the image, respectively. This is
	    accomplished by adding one to the `adjW` and `adjH` parameters of
	    `SpatialFullConvolution`.
	--]]
	local pw, aw, m

	if scale % 2 == 0 then
		pw = scale + scale / 2
		aw = 1
	else
		pw = scale + (scale - 1) / 2
		aw = 0
	end

	m = SpatialFullConvolution(fm_in, fm_out, kw, kw, scale, scale, pw, pw, aw, aw)

	local w, b = m.weight, m.bias
	w:zero()
	b:zero()

	local v = torch.Tensor(inner_width)
	for i = 1, inner_width do v[i] = 1 - math.abs(i - scale) / scale end
	local init = torch.ger(v, v)

	for i = 1, fm_in do
		for j = 1, k do
			local i_out = k * (i - 1) + j
			w[{{i_out}, {i}, {extra_width + 1, extra_width + inner_width},
				{extra_width + 1, extra_width + inner_width}}]:copy(init)
		end
	end

	if extra_width == 0 then
		return nn.Sequential():add(nn.SpatialReplicationPadding(1, 1, 1, 1)):add(m)
	else
		-- XXX: cropping using `SpatialZeroPadding` is very slow.
		return nn.Sequential():
			add(nn.SpatialReplicationPadding(1, 1, 1, 1)):
			add(m):
			add(nn.SpatialZeroPadding(-extra_width, 0, -extra_width, 0))
	end
end

function nninit.make_spatial_blur_conv(args)
	local kw, fm_in, k = validate_common_conv_args(args)
	assert(k == 1)

	local iw = args.input_width
	assert(iw >= kw)

	local std = args.std or 0.25
	assert(std > 0)

	local pad_lt, pad_rb
	if kw % 2 == 0 then pad_lt, pad_rb = (kw - 2) / 2, kw / 2
	else pad_lt, pad_rb = kw / 2, kw / 2 end

	local kernel = image.gaussian({size = kw, normalize = true, sigma = std}):view(1, 1, kw)
	local conv = SpatialConvolution(1, 1, kw, kw)
	conv.bias, conv.gradBias = nil, nil
	conv.weight:copy(kernel)

	return nn.Sequential()
		:add(nn.View(-1, iw, iw))
		:add(nn.SpatialReplicationPadding(pad_lt, pad_rb, pad_lt, pad_rb))
		:add(conv)
		:add(nn.View(-1, fm_in, iw, iw))
end

--[[
Implements the data-driven initialization method described in [1]. Note that the weights of the
network should have already been initialized using a static initialization method. All biases should
be initialized to zero. The method described in [2] corresponds to only performing the intra-layer
initialization in this implementation. Inter-layer initialization can be disabled using the option
`do_inter_layer_init`.

Notes:
* Only transfer the model to the GPU **after** the initialization process.

[1]: http://arxiv.org/pdf/1511.06856.pdf
"Data-dependent initialization of convolutional neural networks"

[2]: http://arxiv.org/abs/1511.06422
"All you need is a good init"

- TODO: extension to non-feedforward architectures? Technically, the method described in [1] only
  applies to feedforward archiectures, but in practice, many models are not purely feedforward.
--]]
function nninit.data_driven(model, eval_func, args)
	args = args or {}

	local beta                = args.beta       or 0
	local dampening           = args.dampening  or 0.5
	local batch_size          = args.batch_size or 128
	local max_iters           = args.max_iters  or 50
	local tol                 = args.tol        or 1e-3
	local verbose             = args.dry_run    or true
	local dry_run             = args.dry_run    or false
	local do_inter_layer_init = args.do_inter_layer_init

	if do_inter_layer_init == nil then do_inter_layer_init = true end

	assert(dampening > 0 and dampening < 1)
	assert(batch_size > 1)
	assert(max_iters > 0)
	assert(tol > 0 and tol < 1)

	local module_info = {}

	for i = 1, #model.modules do
		local module       = model.modules[i]
		local typename     = torch.typename(module)

		-- If this model has already been processed by this function, then all of the
		-- learnable modules will be nested in `nn.Sequential` containers.
		if typename == 'nn.Sequential' and #module.modules == 2 and
		   torch.typename(module.modules[2]) == 'nn.Mul' then
			module = module.modules[1]
			typename = torch.typename(module)
		end

		local w, b         = module.weight, module.bias
		local is_learnable = w ~= nil or b ~= nil

		-- Determine whether we know how to deal with this module type.
		local status, fan_in, _ = pcall(function() return compute_fan(module) end)

		if status then
			local output_groups
			if not typename:find('SpatialFullConvolution') then output_groups = w:size(1)
			else output_groups = w:size(2) end

			assert(w:nElement() == fan_in * output_groups)

			if b ~= nil then
				assert(b:nDimension() == 1)
				assert(b:size(1) == output_groups)
				--assert(torch.all(torch.eq(b, 0)))
			end

			local scaling

			if not dry_run and do_inter_layer_init then
				scaling = nn.Mul()
				scaling.weight[1] = 1
				scaling.accGradParameters = function() end
				if w:type() == 'torch.CudaTensor' then scaling:cuda() end

				model.modules[i] = nn.Sequential():add(module):add(scaling)
			end

			local info = {
				index         = i,
				module        = module,
				scaling       = scaling,
				fan_in        = fan_in,
				output_groups = output_groups,
			}

			table.insert(module_info, info)
		elseif is_learnable then
			print(F"Warning: module {j} of type {typename} has learnable parameters "   ..
				"but is unsupported. It will be ignored during the initialization " ..
				"procedure.")
		end
	end

	if verbose then print("Intra-layer initialization.") end

	-- "Within-layer initialization"
	if not dry_run then
		for i, info in pairs(module_info) do
			eval_func(nil, false)

			local module    = info.module
			local typename  = torch.typename(module)
			local w, b, out = module.weight, module.bias, module.output

			assert(out:nDimension() >= 2)
			assert(out:size(1) == batch_size)
			assert(out:size(2) == info.output_groups)
			local output_group_size = out[{{1}, {1}}]:nElement()

			local out_ = out:view(batch_size, info.output_groups, output_group_size)
				:transpose(1, 2)
				:clone()
				:view(info.output_groups, batch_size * output_group_size)

			local mu, sigma = out_:mean(2), out_:std(2)

			--[[
			For linear layers, this divides each row of the weight matrix by the variance of the
			corresponding activation. For convolutional layers, this divides the weight
			corresponding to each output feature map by the varince of the activations in that
			output feature map.
			--]]

			if not typename:find('SpatialFullConvolution') then
				local w_     = w:view(info.output_groups, info.fan_in)
				local sigma_ = torch.expand(sigma:view(info.output_groups, 1), info.output_groups, info.fan_in)
				w_:cdiv(sigma_)
			else
				local s1, s3 = w:size(1), w[{{1}, {1}}]:nElement()
				local w_     = w:view(s1, info.output_groups, s3)
				local sigma_ = torch.expand(sigma:view(1, info.output_groups, 1), s1, info.output_groups, s3)
				w_:cdiv(sigma_)
			end

			if b ~= nil and output_group_size ~= 1 then b:fill(beta):addcdiv(b, -1, mu, sigma) end
			collectgarbage()

			if verbose then xlua.progress(i, #module_info) end
		end
	end

	if verbose then
		for i, info in pairs(module_info) do
			eval_func(nil, false)

			local module    = info.module
			local w, b, out = module.weight, module.bias, module.output
			print(F"Layer {info.index}. Output mean: {out:mean()}; output std: {out:std()}.")
		end
	end

	if not do_inter_layer_init then return end

	local rates = torch.Tensor(#module_info)
	local ratios = torch.Tensor(#module_info)
	local avg_rate, converged_layers = 0, 0

	if verbose then print("\nInter-layer initialization.") end

	-- "Between-layer initialization"
	(function() for i = 1, max_iters do
		rates:zero()

		for j = 1, batch_size do
			eval_func(j, true)

			for k, info in pairs(module_info) do
				local gw = info.module.gradWeight
				rates[k] = rates[k] + gw:norm()
			end
		end

		for j, info in pairs(module_info) do
			rates[j] = rates[j] / (batch_size * info.module.weight:norm())
		end

		avg_rate = rates:prod()^(1 / rates:size(1))
		converged_layers = 0

		for j, info in pairs(module_info) do
			local module, scaling = info.module, info.scaling
			local w, b = module.weight, module.bias

			local r = avg_rate / rates[j]
			if math.abs(r - 1) < tol then converged_layers = converged_layers + 1 end

			r = r^dampening
			ratios[j] = r

			if not dry_run then
				w:div(r)
				if b ~= nil then b:div(r) end
				scaling.weight[1] = r
			end
		end

		if dry_run or converged_layers == #module_info then return end
		if verbose then xlua.progress(i, max_iters) end
	end end)()

	if verbose or converged_layers ~= #module_info then
		if not dry_run and converged_layers ~= #module_info then
			print(F"Warning: {#module_info - converged_layers} / {#module_info} layers did "  ..
				F"not satisfy the specified tolerance ({tol}) for the inter-layer change " ..
				F"rate ratio within the specified number of iterations ({max_iters}).")
		end

		print(F"Geometric average of change rates across network: {avg_rate}.")

		for i, info in pairs(module_info) do
			local module      = info.module
			local typename    = torch.typename(module)
			local rate, ratio = rates[i], ratios[i]^(1 / dampening)

			print(F"Layer {info.index} ({typename}). Change rate: {rate}; change ratio: {ratio}.")
		end
	end
end

return nninit
