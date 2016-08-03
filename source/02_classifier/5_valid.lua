
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'cunn'

------------------------------
-- function
------------------------------

function valid()
	collectgarbage()
	print("")

	-- Load valid images
	print(" => Load valid images")
	path.mkdir("preprocess")
	local prepro_list = getFilename("preprocess")
	local seed = opt.seed

	assert(contains(prepro_list, "valid_data_s" .. seed .. ".set"))
	local valid_data = torch.load("preprocess/valid_data_s" .. seed .. ".set")

	assert(contains(prepro_list, "valid_data_s" .. seed .. ".set"))
	local valid_label = torch.load("preprocess/valid_label_s" .. seed .. ".set")

	assert(contains(prepro_list, "trfolders.set"))
	local trfolders = torch.load("preprocess/trfolders.set")

	for i=1,3 do
		valid_data[{{},i,{},{}}]:add(-res_mean[i]):div(res_std[i])
	end
	
	collectgarbage()

	local valid_confusion = optim.ConfusionMatrix(trfolders)

	-- local vars
	local valid_size = valid_data:size(1)
	valid_score = 0

	-- set model to evaluate mode
	model:evaluate()

	-- test over test data
	print(sys.COLORS.green .. '==> validating on valid set:')

	for t = 1,valid_size,opt.batchSize do

		-- disp progress
		xlua.progress(t, valid_size)

		local local_batchSize = math.min(opt.batchSize, valid_size - t + 1)
		local inputs = torch.Tensor(local_batchSize, 3, 224, 224)
		local targets = torch.Tensor(local_batchSize)

		for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do

			-- get new sample
			local img_ = valid_data[i]

			inputs[{local_count}] = crop_imgs(img_)
			targets[{local_count}] = valid_label[i]
		end

		inputs = inputs:cuda()
		targets = targets:cuda()
		
		local preds = model:forward(inputs)
		collectgarbage()

		-- calc valid score
		local err = criterion:forward(preds, targets)
		valid_score = valid_score + err * local_batchSize
		valid_confusion:batchAdd(preds, targets)
	end
	xlua.progress(valid_size, valid_size)

	-- print valid score
	valid_score = valid_score / valid_size
	print('\tvalid_score: ' .. string.format("%.4f", valid_score))

	-- print total valid
	valid_confusion:updateValids()
	print('\tvalid_accuracy: ' .. string.format("%.1f", valid_confusion.totalValid * 100) .. " [%]")
end
