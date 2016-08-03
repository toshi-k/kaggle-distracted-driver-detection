
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

function load_with_mask(file_path, mask_path)

	local img = image.load(file_path)
	local mask = image.load(mask_path)[1]

	mask = image.scale(mask, 640)
	mask = mask[{{81,560},{}}]

	local mask_sum1 = torch.sum(mask, 1)[1]
	local x1, x2 = get_mergin(mask_sum1)

	local mask_sum2 = torch.sum(mask, 2)[{{},1}]
	local y1, y2 = get_mergin(mask_sum2)

	img = img[{{},{y1, y2},{x1, x2}}]
	local img_resize = myresize(img, 224, 260)

	return img_resize
end

function stack_imgs(tb_images, tb_labels, is_train, name_head, name_tail)
	local nrow = #tb_images

	local label_pre
	local data_pre = torch.Tensor(nrow, 3, 260, 260):fill(0.5)
	if is_train then label_pre = torch.Tensor(nrow) end

	for i, file_path in pairs(tb_images) do

		if i%100 == 0 then
			xlua.progress(i, nrow)
			collectgarbage()
		end

		local mask_path = file_path
		mask_path = string.gsub(file_path, "train", "train_mask_pred")
		mask_path = string.gsub(mask_path, "test", "test_mask_pred")

		local img = load_with_mask(file_path, mask_path)
		data_pre[{{i}}] = img

		if is_train then label_pre[i] = tb_labels[i] end
	end
	xlua.progress(nrow, nrow)

	torch.save("preprocess/" .. name_head .. "data" .. name_tail .. ".set", data_pre)
	if is_train then
		torch.save("preprocess/" .. name_head .. "label" .. name_tail .. ".set", label_pre)
	end
end

function stack_train_imgs()

	path.mkdir("preprocess")
	local prepro_list = getFilename("preprocess")
	local trfolders = getFilename( "../../dataset/train" )
	local seed = opt.seed

	if not (contains(prepro_list, "train_data_s" .. seed .. ".set") and
			contains(prepro_list, "train_label_s" .. seed .. ".set") and
			contains(prepro_list, "valid_data_s" .. seed .. ".set") and
			contains(prepro_list, "valid_label_s" .. seed .. ".set")) then

		local tb_train_images = {}
		local tb_train_labels = {}
		local tb_valid_images = {}
		local tb_valid_labels = {}

		for i, foname in ipairs(trfolders) do
			local trImages = getFilename( "../../dataset/train/" .. foname .. "/")
			for j, imgname in ipairs(trImages) do
				local path = "../../dataset/train/" .. foname .. "/" .. imgname
				if contains(driver_list_train, image2driver[imgname]) then
					table.insert(tb_train_images, path)
					table.insert(tb_train_labels, i)
				else
					table.insert(tb_valid_images, path)
					table.insert(tb_valid_labels, i)
				end
			end
		end

		print("==> stack train images")
		stack_imgs(tb_train_images, tb_train_labels, true, "train_", "_s" .. seed)
		print("==> stack valid images")
		stack_imgs(tb_valid_images, tb_valid_labels, true, "valid_", "_s" .. seed)
		collectgarbage()
	end

	if not contains(prepro_list, "trfolders.set") then
		torch.save("preprocess/trfolders.set", trfolders)
		collectgarbage()
	end
end

function stack_test_imgs()

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

			stack_imgs(tb_test_images, _, false, "test_", "_" .. test_set_id)
			collectgarbage()
		end
	end

	assert(num_loadimg==test_nrow or num_loadimg==0)

	if not contains(prepro_list, "teimages.set") then
		torch.save("preprocess/teimages.set", teImages)
		collectgarbage()
	end
end

------------------------------
-- main
------------------------------

image2driver, driver_list_train, driver_list_valid = read_driver_list()
