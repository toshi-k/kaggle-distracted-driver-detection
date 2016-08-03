
------------------------------
-- library
------------------------------

require 'torch'
require 'image'

------------------------------
-- function
------------------------------

require 'lib/getfilename'
require 'lib/resize'
require 'lib/contains'
require 'lib/unique'

function read_driver_list()

	math.randomseed(opt.seed)
	torch.manualSeed(opt.seed)
	cutorch.manualSeed(opt.seed)

	local image2driver = {}
	local driver_list = {}
	local fp = io.open("../../dataset/driver_imgs_list.csv", "r")

	-- remove header
	fp:read()
	
	while true do
		local line = fp:read()
		if line == nil then 
			break 
		else
			line = string.gsub(line,'\r',"" )
			local line_tb = utils.split(line, ",")
			image2driver[line_tb[3]] = line_tb[1]
			table.insert(driver_list, line_tb[1])
		end
	end
	fp:close()

	driver_list = unique(driver_list)

	local driver_list_train = {}
	local driver_list_valid = {}
	local shuffle = torch.randperm(#driver_list)

	for i=1,4 do
		table.insert(driver_list_valid, driver_list[shuffle[i]])
	end
	for i=5,#driver_list do
		table.insert(driver_list_train, driver_list[shuffle[i]])
	end

	return image2driver, driver_list_train, driver_list_valid
end

function read_class_list()

	path.mkdir("preprocess")
	local prepro_list = getFilename("preprocess")
	local trfolders = getFilename( "../../dataset/train" )
	local seed = opt.seed

	local image2class = {}

	for i, foname in ipairs(trfolders) do
		local trImages = getFilename( "../../dataset/train/" .. foname .. "/")
		for j, imgname in ipairs(trImages) do
			image2class[imgname] = foname
		end
	end

	return image2class
end

function stack_imgs(tb_images, tb_labels, is_train, name_data, name_label, fill_value)
	local nrow = #tb_images

	local label_pre
	local data_pre = torch.Tensor(nrow, 3, 260, 260):fill(fill_value)
	if is_train then label_pre = torch.Tensor(nrow) end

	for i, file_path in pairs(tb_images) do

		if i%10 == 0 then
			xlua.progress(i, nrow)
			collectgarbage()
		end

		local img = image.load(file_path)
		img = img:index(1, torch.LongTensor{3, 2, 1})
		data_pre[{{i}}] = myresize(img, 224, 260, fill_value)
		if is_train then label_pre[i] = tb_labels[i] end
	end
	xlua.progress(nrow, nrow)

	torch.save("preprocess/" .. name_data .. ".set", data_pre)
	if is_train then
		torch.save("preprocess/" .. name_label .. ".set", label_pre)
	end
end

function stack_train_imgs()

	path.mkdir("preprocess")
	local prepro_list = getFilename("preprocess")
	local trfolders = getFilename( "../../dataset/train_mask")
	local seed = opt.seed

	if not (contains(prepro_list, "train_data_s" .. seed .. ".set") and
			contains(prepro_list, "train_target_s" .. seed .. ".set") and
			contains(prepro_list, "valid_data_s" .. seed .. ".set") and
			contains(prepro_list, "valid_target_s" .. seed .. ".set")) then

		local tb_train_images = {}
		local tb_train_targets = {}
		local tb_valid_images = {}
		local tb_valid_targets = {}

		for i, mask_name in ipairs(trfolders) do
			local image_name = string.sub(mask_name, 1, -11) .. ".jpg"
			local path_image = "../../dataset/train/" .. image2class[image_name] .. "/" .. image_name
			local path_target = "../../dataset/train_mask/" .. mask_name
			if contains(driver_list_train, image2driver[image_name]) then
				table.insert(tb_train_images, path_image)
				table.insert(tb_train_targets, path_target)
			else
				table.insert(tb_valid_images, path_image)
				table.insert(tb_valid_targets, path_target)
			end
		end

		print("==> stack train images")
		stack_imgs(tb_train_images, _, false, "train_data_s" .. seed, _, 0.5)
		print("==> stack train masks")
		stack_imgs(tb_train_targets, _, false, "train_target_s" .. seed, _, 0)
		print("==> stack valid images")
		stack_imgs(tb_valid_images, _, false, "valid_data_s" .. seed, _, 0.5)
		print("==> stack valid masks")
		stack_imgs(tb_valid_targets, _, false, "valid_target_s" .. seed, _, 0)
		collectgarbage()
	end

	if not contains(prepro_list, "trfolders.set") then
		torch.save("preprocess/trfolders.set", trfolders)
		collectgarbage()
	end
end

function stack_test_imgs_test()

	path.mkdir("preprocess")
	local prepro_list = getFilename("preprocess")

	local teImages = getFilename( "../../dataset/test")
	local test_nrow = #teImages

	local num_loadimg = 0
	for test_set_id, I in ipairs( tablex.range(1, test_nrow, 20000) ) do
		if not contains(prepro_list, "test_data_" .. test_set_id ..".set") then

			print("==> stack test images [" .. test_set_id .. "]")

			local tb_test_images = {}
			for i = I,math.min(I+20000-1,test_nrow) do
				local file_name = "../../dataset/test/" .. teImages[i]
				table.insert(tb_test_images, file_name)
				num_loadimg = num_loadimg + 1
			end

			stack_imgs(tb_test_images, _, false, "test_data_" .. test_set_id, _, 0.5)
			collectgarbage()
		end
	end

	assert(num_loadimg==test_nrow or num_loadimg==0)

	if not contains(prepro_list, "teimages.set") then
		torch.save("preprocess/teimages.set", teImages)
		collectgarbage()
	end
end

function stack_test_imgs_train()

	trfolders = getFilename( "../../dataset/train" )

	path.mkdir("preprocess")
	local prepro_list = getFilename("preprocess")

	local tb_train_images = {}
	local teImages = {}

	for i, foname in ipairs(trfolders) do
		local trImages = getFilename( "../../dataset/train/" .. foname .. "/")
		for j, imgname in ipairs(trImages) do
			table.insert(tb_train_images, "../../dataset/train/" .. foname .. "/" .. imgname)
			table.insert(teImages, foname .. "/" .. imgname)
		end
	end

	if not contains(prepro_list, "train_data_all.set") then
		stack_imgs(tb_train_images, _, false, "train_data_all", _, 0.5)
	end

	if not contains(prepro_list, "trimages.set") then
		torch.save("preprocess/trimages.set", teImages)
		collectgarbage()
	end
end

------------------------------
-- main
------------------------------

image2driver, driver_list_train, driver_list_valid = read_driver_list()
image2class = read_class_list()
