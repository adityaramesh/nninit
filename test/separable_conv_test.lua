local nn = require('nn')
local image = require('image')

local v = torch.Tensor{1 / 2, 1, 1 / 2}
local k = torch.ger(v, v)

local img = image.lena()
img = image.scale(img, 128, 128)

local scale = 2
local support = 1

local kw = 3
local pw = (kw + 1) / 2 + scale / 2 + scale * (support - 1)
local aw = 1

local conv_xy = nn.SpatialFullConvolution(1, 1, kw, kw, scale, scale, pw, pw, aw, aw)
conv_xy.weight[{{1}, {1}}]:copy(k)
conv_xy.bias = nil

local m1 = nn.Sequential()
	:add(nn.View(3, 1, 128, 128))
	:add(nn.SpatialReplicationPadding(support, support, support, support))
	:add(conv_xy)
	:add(nn.View(3, 256, 256))

local out_1 = m1:forward(img)
image.save('upsamp_1.png', out_1)

local conv_x = nn.SpatialFullConvolution(1, 1,   kw, 1,   scale, 1,   pw, 0,   aw, 0)
local conv_y = nn.SpatialFullConvolution(1, 1,   1, kw,   1, scale,   0, pw,   0, aw)

conv_x.weight[{{1}, {1}}]:copy(v)
conv_x.bias = nil

conv_y.weight[{{1}, {1}}]:copy(v)
conv_y.bias = nil

local m2 = nn.Sequential()
	:add(nn.View(3, 1, 128, 128))
	:add(nn.SpatialReplicationPadding(support, support, support, support))
	:add(conv_x)
	:add(conv_y)
	:add(nn.View(3, 256, 256))

local out_2 = m2:forward(img)
image.save('upsamp_2.png', out_2)

print(torch.norm(torch.add(out_1, -1, out_2)))

sys.tic()
for i = 1, 100 do m1:forward(img) end
print(sys.toc())

sys.tic()
for i = 1, 100 do m2:forward(img) end
print(sys.toc())
