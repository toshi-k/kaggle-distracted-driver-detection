
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
	path.mkdir("../../dataset/test_mask_pred")

	local prepro_list = getFilename("preprocess")
	
	assert(contains(prepro_list, "teimages.set"))
	local teImages = torch.load("preprocess/teimages.set")

	assert(contains(prepro_list, "trfolders.set"))
	local trfolders = torch.load("preprocess/trfolders.set")

	-- set model to evaluate mode
	model:evaluate()

	local gaussian = image.gaussian(3,3)
	local blur = nn.SpatialConvolution(1,1,3,3,1,1,1,1):cuda()
	gaussian:div(torch.sum(gaussian) * 0.5)
	blur.weight[1] = gaussian
	blur.bias:zero()
	blur:cuda()

	-- test over test data
	print(sys.COLORS.red .. '==> testing on test set:')
	local test_nrow = #teImages
	
	for test_set_id, I in ipairs( tablex.range(1, test_nrow, 20000) ) do

		-- Load test images
		print(" => Load test images")
		assert(contains(prepro_list, "test_data_" .. test_set_id ..".set"))
		local test_data = torch.load("preprocess/" .. "test_data_" .. test_set_id ..".set")

		-- Preprocessing
		test_data:mul(255)
		for i=1,3 do
			test_data[{{},i,{},{}}]:add(-vgg_mean[i])
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
				-- img_ = image.scale(img_, 260, 260)

				inputs[{local_count}] = crop_imgs(img_)
			end

			inputs = inputs:cuda()
			local preds = model:forward(inputs)
			collectgarbage()

			for local_count, i in ipairs( tablex.range(t, t+local_batchSize-1) ) do

				local save_image_smooth1 = blur1:forward(preds[{{local_count}}])[1]
				local save_image_smooth2 = blur2:forward(preds[{{local_count}}])[1]

				local original_image = inputs[local_count]
				for i=1,3 do
					original_image[{{i},{},{}}]:add(vgg_mean[i])
				end
				original_image:div(255)
				original_image = original_image:index(1, torch.LongTensor{3, 2, 1})

				local masked_image = original_image:clone()

				original_image:add(-0.5)
				local mask_reshpae1 = torch.repeatTensor(save_image_smooth1,3,1,1)
				local mask_reshpae2 = torch.repeatTensor(save_image_smooth2,3,1,1)
				local masked_image1 = torch.cmul(original_image, mask_reshpae1)
				local masked_image2 = torch.cmul(original_image, mask_reshpae2)
				local masked_image = original_image * 0.1 + masked_image1 * 0.7 + masked_image2 * 0.2
				masked_image:add(0.5)

				masked_image = mycrop(masked_image:float(), 260, 0.5)
				local save_name = paths.concat("../../dataset", "test_mask_pred", teImages[i])
				path.mkdir(sys.dirname(save_name))
				image.save(save_name, mask_reshpae2)
			end
		end
	end

	xlua.progress(test_nrow, test_nrow)
end
