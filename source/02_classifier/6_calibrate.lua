
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'cunn'

------------------------------
-- function
------------------------------

function get_mean_prob(label)
	local mean_prob = torch.Tensor(10):fill(0)
	local num_img = label:size(1)

	for i = 1,num_img do
		mean_prob[{{label[i]}}]:add(1)
	end

	mean_prob:div(label:size(1))

	return mean_prob
end

function calibrate()
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

	collectgarbage()

	local valid_confusion = optim.ConfusionMatrix(trfolders)

	local num_sample = valid_data:size(1)
	local valid_size = num_sample

	local valid_score = 0

	local mean_prob = get_mean_prob(valid_label)

	-- set model to evaluate mode
	model:evaluate()

	local mean_output = torch.Tensor(10):fill(0):cuda()

	-- calibration over valid data
	print(sys.COLORS.yellow .. '==> calibration on valid set:')

	for t = 1,valid_size,opt.batchSize do
		-- disp progress
		xlua.progress(t, valid_size)

		-- create mini batch
		local local_batchSize = math.min(opt.batchSize, valid_size - t + 1)
		local inputs = torch.Tensor(local_batchSize, 3, 224, 224)
		local targets = torch.Tensor(local_batchSize)

		for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do

			local img = valid_data[i]

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

			inputs[{local_count}] = crop_imgs(img)
			targets[{local_count}] = valid_label[i]
		end

		inputs = inputs:cuda()
		targets = targets:cuda()

		-- Preprocessing
		for i=1,3 do
			inputs[{{},i,{},{}}]:add(-res_mean[i]):div(res_std[i])
		end
		collectgarbage()

		local log_output = model:forward(inputs)
		local output = torch.exp(log_output)

		local err = criterion:forward(log_output, targets)
		valid_score = valid_score + err * local_batchSize
		valid_confusion:batchAdd(log_output, targets)
 
		mean_output:add(output:sum(1))
	end
	xlua.progress(valid_size, valid_size)

	valid_score = valid_score / valid_size
	mean_output:div(valid_size)
	
	mean_output = mean_output:float()
	local logit_mean_prob = torch.log(mean_prob) - torch.log(torch.add(-mean_prob,1))
	local logit_mean_out = torch.log(mean_output) - torch.log(torch.add(-mean_output,1))

	local diff = logit_mean_prob - logit_mean_out
	model:get(15):updateBias(diff:cuda())

	local diff_score = diff:pow(2):sum()

	print('\tvalid_score: ' .. string.format("%.4f", valid_score) .. " (diff: " .. string.format("%.4f", diff_score) .. ")" )

	-- print total valid
	valid_confusion:updateValids()
	print('\tvalid_accuracy: ' .. string.format("%.1f", valid_confusion.totalValid * 100) .. " [%]")
end
