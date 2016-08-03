
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
cmd:option('-batchSize', 6, 'mini-batch size')
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
dofile "6_test_train.lua"

-- training
vgg_mean = {103.939, 116.779, 123.68}

stack_train_imgs()
stack_test_imgs_train()
stack_test_imgs_test()

print '==> training!'
for Itr = 1,20 do

	-- train
	train()

	-- valid
	valid()
end

-- test
test()
test_train()
