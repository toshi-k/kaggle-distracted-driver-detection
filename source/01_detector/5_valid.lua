
------------------------------
-- library
------------------------------

require 'torch'
require 'xlua'
require 'optim'
require 'cunn'
require 'image'

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
	local valid_target = torch.load("preprocess/valid_target_s" .. seed .. ".set")

	valid_data:mul(255)
	for i=1,3 do
		valid_data[{{},i,{},{}}]:add(-vgg_mean[i])
	end
	
	collectgarbage()

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
		local targets = torch.Tensor(local_batchSize, 1, 224, 224)

		for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do

			-- get new sample
			local img_ = valid_data[i]
			local target_ = valid_target[i]

			inputs[{local_count}] = crop_imgs(img_)
			targets[{local_count}] = crop_imgs(target_)[1]
		end

		inputs = inputs:cuda()
		targets = targets:cuda()
		
		local preds = model:forward(inputs)

		path.mkdir("_save")
		for s=1,preds:size(1) do
			local save_image = preds[s]
			image.save("_save/" .. tostring(t+s-1) .. "_mask.png", save_image)

			local save_image_smooth1 = blur1:forward(preds[{{s}}])[1]
			image.save("_save/" .. tostring(t+s-1) .. "_mask_smooth_1.png", save_image_smooth1)

			local save_image_smooth2 = blur2:forward(preds[{{s}}])[1]
			image.save("_save/" .. tostring(t+s-1) .. "_mask_smooth_2.png", save_image_smooth2)

			local original_image = inputs[s]
			for i=1,3 do
				original_image[{{i},{},{}}]:add(vgg_mean[i])
			end
			original_image:div(255)
			original_image = original_image:index(1, torch.LongTensor{3, 2, 1})
			image.save("_save/" .. tostring(t+s-1) .. "_original.png", original_image)

			original_image:add(-0.5)
			local mask_reshpae1 = torch.repeatTensor(save_image_smooth1,3,1,1)
			local mask_reshpae2 = torch.repeatTensor(save_image_smooth2,3,1,1)
			local masked_image1 = torch.cmul(original_image, mask_reshpae1)
			local masked_image2 = torch.cmul(original_image, mask_reshpae2)
			local masked_image = original_image * 0.1 + masked_image1 * 0.7 + masked_image2 * 0.2
			masked_image:add(0.5)
			image.save("_save/" .. tostring(t+s-1) .. "_masked.png", masked_image)
		end
		collectgarbage()

		-- calc valid score
		local err = criterion:forward(preds, targets)
		valid_score = valid_score + err * local_batchSize
	end
	xlua.progress(valid_size, valid_size)

	-- print valid score
	valid_score = valid_score / valid_size
	print("\tvalid_score: " .. string.format("%.4f", valid_score))
end
