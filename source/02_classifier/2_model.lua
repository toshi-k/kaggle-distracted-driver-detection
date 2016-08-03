
------------------------------
-- library
------------------------------

require 'torch'
require 'nn'
require 'cutorch'
require 'cunn'
require 'cudnn'

require "lib/Calibrator"

------------------------------
-- function
------------------------------

function newmodel()

	-- 10-class problem
	local noutputs = 10

	-- load pretrain model
	local model_dir = 'pretrain'
	local model = torch.load(paths.concat(model_dir, 'resnet-200.t7'))

	-- remove non-used modules
	model:remove(14)

	-- add output layers
	model:add(nn.Linear(2048, noutputs))
	model:add(nn.Calibrator(noutputs))
	model:add(nn.LogSoftMax())
	
	return model
end

------------------------------
-- main
------------------------------

-- model
model = newmodel()
-- print(model)

-- criterion
criterion = nn.ClassNLLCriterion()
print '==> here is the loss function:'
print(criterion)
