
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'cunn'

------------------------------
-- function
------------------------------

function test()
	collectgarbage()
	print("")

	path.mkdir("preprocess")
	local prepro_list = getFilename("preprocess")
	
	assert(contains(prepro_list, "teimages.set"))
	local teImages = torch.load("preprocess/teimages.set")

	assert(contains(prepro_list, "trfolders.set"))
	local trfolders = torch.load("preprocess/trfolders.set")

	-- set model to evaluate mode
	model:evaluate()

	local test_name = "submission_pre"
	path.mkdir("../../submission/" .. test_name)

	local file_name = test_name .. "_s" .. opt.seed ..
						"_train" .. string.format("%.4f", train_score) .. 
						"_valid" .. string.format("%.4f", valid_score) .. ".csv"
	local path = paths.concat("../../submission", test_name, file_name)
	local fp = io.open(path, "w")

	local headers = {"img"}
	for i, var in pairs(trfolders) do
		table.insert(headers, var)
	end
	local headwrite = table.concat(headers, ",")
	fp:write(headwrite.."\n")

	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')
	local test_nrow = #teImages
	
	for test_set_id, I in ipairs( tablex.range(1, test_nrow, 20000) ) do

		-- Load test images
		print(" => Load test images")
		assert(contains(prepro_list, "test_data_" .. test_set_id ..".set"))
		local test_data = torch.load("preprocess/" .. "test_data_" .. test_set_id ..".set")

		-- Preprocessing
		for i=1,3 do
			test_data[{{},i,{},{}}]:add(-res_mean[i]):div(res_std[i])
		end

		local test_set_endpoint = math.min(I+20000-1, test_nrow)
		for t = I,test_set_endpoint,opt.batchSize do

			-- disp progress
			xlua.progress(t, test_nrow)

			local local_batchSize = math.min(opt.batchSize, test_set_endpoint - t + 1)
			local inputs = torch.Tensor(local_batchSize, 3, 224, 224)

			for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do

				-- get new sample
				local img_ = test_data[i-I+1]

				inputs[{local_count}] = crop_imgs(img_)
			end

			inputs = inputs:cuda()
			local preds = model:forward(inputs):float()
			preds:exp()
			collectgarbage()

			for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do
				local row = {teImages[i]}

				for class_id = 1, #trfolders do
					table.insert(row, tostring(preds[{local_count, class_id}]))
				end
				local rowwrite = table.concat(row, ",")
				fp:write(rowwrite.."\n")
			end
		end
	end

	xlua.progress(test_nrow, test_nrow)
	fp:close()
end
