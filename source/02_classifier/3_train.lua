
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'cunn'

------------------------------
-- main
------------------------------

-- cuda:
model:cuda()
criterion:cuda()

-- Retrieve parameters and gradients:
if model then
	parameters,gradParameters = model:getParameters()
end

print '==> configuring optimizer'
-- SGD
optimState = {
	learningRate = opt.learningRate,
	weightDecay = opt.weightDecay,
	momentum = opt.momentum,
	learningRateDecay = 1e-7
}
optimMethod = optim.adam

------------------------------
-- function
------------------------------

function train()
	collectgarbage()
	print("")

	-- epoch tracker
	epoch = epoch or 1

	-- Load train images
	print(" => Load train images")
	path.mkdir("preprocess")
	local prepro_list = getFilename("preprocess")
	local seed = opt.seed

	assert(contains(prepro_list, "train_data_s" .. seed .. ".set"))
	local train_data = torch.load("preprocess/train_data_s" .. seed .. ".set")

	assert(contains(prepro_list, "train_label_s" .. seed .. ".set"))
	local train_label = torch.load("preprocess/train_label_s" .. seed .. ".set")

	assert(contains(prepro_list, "trfolders.set"))
	local trfolders = torch.load("preprocess/trfolders.set")

	-- This matrix records the current confusion across classes
	local confusion = optim.ConfusionMatrix(trfolders)

	local num_sample = train_data:size(1)
	local train_size = num_sample

	-- set model to training mode
	model:training()

	-- shuffle at each epoch
	local shuffle = torch.randperm(train_size)

	train_score = 0

	-- do one epoch
	print(sys.COLORS.cyan .. '==> training on train set: # ' .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
	for t = 1,train_size,opt.batchSize do

		-- disp progress
		xlua.progress(t, train_size)

		local local_batchSize = math.min(opt.batchSize, train_size - t + 1)
		local inputs = torch.Tensor(local_batchSize, 3, 224, 224)
		local targets = torch.Tensor(local_batchSize)

		-- create mini batch
		for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do

			-- load new sample
			local id = shuffle[i]

			local img = train_data[(id-1)%num_sample+1]
			local target = train_label[(id-1)%num_sample+1]

			-- [1] rotate
			local r = math.random()*2-1
			img = image.rotate(img, r*math.pi*0.05, "bilinear")

			-- [2] scaling
			local s = 20 * (math.random(3)-2)
			img = image.scale(img, img:size(2) + s, img:size(3) + s)

			-- [3] translate
			local x = (math.random()*2 - 1) * 10
			local y = (math.random()*2 - 1) * 10
			img = image.translate(img, x, y)

			-- [4] gamma
			local g = (math.random()*2 - 1) * 0.2 + 1
			img:pow(g)

			inputs[{{local_count}}] = crop_imgs(img)
			targets[{local_count}] = target
		end

		-- Preprocessing
		for i=1,3 do
			inputs[{{},i,{},{}}]:add(-res_mean[i]):div(res_std[i])
		end
		collectgarbage()

		inputs = inputs:cuda()
		targets = targets:cuda()

		-- create closure to evaluate f(X) and df/dX
		local feval = function(x)
			-- get new parameters
			if x ~= parameters then
				parameters:copy(x)
			end

			-- reset gradients
			gradParameters:zero()

			-- estimate output
			local output = model:forward(inputs)

			-- f is the average of all criterions
			local err = criterion:forward(output, targets)
			local f = err

			-- estimate df/dW
			local df_do = criterion:backward(output, targets)
			model:backward(inputs, df_do)

			-- calc train score
			train_score = train_score + err * local_batchSize
			confusion:batchAdd(output, targets)

			-- normalize gradients
			gradParameters:div(local_batchSize)

			-- return f and df/dX
			return f, gradParameters
		end

		-- optimize on current mini-batch
		optimMethod(feval, parameters, optimState)
		collectgarbage()

	end
	xlua.progress(train_size, train_size)

	-- print train score
	train_score = train_score / train_size
	print('\ttrain_score: ' .. string.format("%.4f", train_score))

	-- print total valid
	confusion:updateValids()
	print('\ttrain_accuracy: ' .. string.format("%.1f", confusion.totalValid * 100) .. " [%]")

	-- save current net
	local filename = paths.concat(opt.save_models, 'model_' .. epoch .. '.net')
	path.mkdir(opt.save_models)

	-- print('\tsaving model to '..filename)
	torch.save(filename, model:clearState())

	epoch = epoch + 1
end
