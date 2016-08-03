
------------------------------
-- library
------------------------------

require 'torch'
require 'image'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'loadcaffe'

------------------------------
-- function
------------------------------

function newmodel()

	-- load pretrain model
	local model_dir = 'pretrain'
	local deploy_text = paths.concat(model_dir, 'deploy.prototxt')
	local model_file = paths.concat(model_dir,'VGG_ILSVRC_16_layers.caffemodel')
	local model = loadcaffe.load(deploy_text, model_file, 'cudnn')

	-- remove non-used modules
	for i=40,32,-1 do
		model:remove(i)
	end

	-- add output layers
	model:add(nn.SpatialFullConvolution(512,1,32,32,32,32))
	model:add(nn.Sigmoid())
	
	return model
end

function newblur(size)
	local model = nn.Sequential()

	local gaussian = image.gaussian(size,size)
	local pad = math.floor(size / 2)
	local blur = nn.SpatialConvolution(1,1,size,size,1,1,pad,pad):cuda()
	gaussian:div(torch.sum(gaussian) * 0.5)
	blur.weight[1] = gaussian
	blur.bias:zero()

	model:add(blur)

	model:add(nn.ReLU())
	model:add(nn.AddConstant(-1,true))
	model:add(nn.MulConstant(-1,true))
	model:add(nn.ReLU())
	model:add(nn.MulConstant(-1,true))
	model:add(nn.AddConstant(1,true))

	return model
end

------------------------------
-- main
------------------------------

-- model
model = newmodel()
-- print(model)

-- soft blur
blur1 = newblur(9)
blur1:cuda()

-- hard blur
blur2 = newblur(19)
blur2:cuda()

-- criterion
criterion = nn.MSECriterion()
print '==> here is the loss function:'
print(criterion)
