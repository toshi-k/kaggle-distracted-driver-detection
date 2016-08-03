
------------------------------
-- library
------------------------------

require 'nn'
require 'cunn'
require 'gnuplot'

------------------------------
-- settings
------------------------------

cmd = torch.CmdLine()
cmd:text('Options:')
-- global:
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-threads', 4, 'number of threads')
-- paths:
cmd:option('-save_models', 'models', 'subdirectory to save/log experiments in')
-- training:
cmd:option('-learningRate', 1e-5, 'learning rate at t=0')
cmd:option('-batchSize', 3, 'mini-batch size')
cmd:option('-weightDecay', 0, 'weight decay')
cmd:option('-momentum', 0, 'momentum')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
math.randomseed(opt.seed)
torch.manualSeed(opt.seed)
cutorch.manualSeed(opt.seed)

------------------------------
-- main
------------------------------

-- disp dataset index:
print '===> dataset'
print("\t   opt.seed:\t" .. opt.seed)

dofile "1_data.lua"
dofile "2_model.lua"
dofile "3_train.lua"
dofile "4_test.lua"
dofile "5_valid.lua"
dofile "6_calibrate.lua"

-- training
res_mean = { 0.485, 0.456, 0.406 }
res_std = { 0.229, 0.224, 0.225 }

stack_train_imgs()
stack_test_imgs()

print '==> training!'
for Itr = 1,3 do

	-- train
	train()

	-- calibration
	valid()
	for c = 1,10 do
		calibrate()
	end

	-- valid
	valid()

	if Itr > 1 then
		optimState.learningRate = optimState.learningRate * 0.2
	end
end

-- test
test()
