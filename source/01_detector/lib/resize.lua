
------------------------------
-- function
------------------------------

function mycrop(img, rang, fill_value)
	local imsize = img:size()
	local x, y = 3, 2

	local crimg1, crimg2

	if imsize[y] > rang then
		crimg1 = img[{{}, {math.floor(imsize[y]/2)-rang/2+1,math.floor(imsize[y]/2)+rang/2},{}}]
	else
		crimg1 = img
	end

	if imsize[x] > rang then
		crimg2 = crimg1[{{}, {}, {math.floor(imsize[x]/2)-rang/2+1,math.floor(imsize[x]/2)+rang/2}}]
	else
		crimg2 = crimg1
	end

	local imsize2 = crimg2:size()
	local ret = torch.Tensor(img:size(1), rang, rang):fill(fill_value)
	ret[{{}, {rang/2+1-math.floor(imsize2[y]/2),rang/2+math.ceil(imsize2[y]/2)},{rang/2+1-math.floor(imsize2[x]/2),rang/2+math.ceil(imsize2[x]/2)}}] = crimg2

	return ret
end

function myresize(img, rang1, rang2, fill_value)
	if (img:size(1) < rang2) and (img:size(2) < rang2) then
		local ret = mycrop(img, rang2, fill_value)
		return ret
	else
		local img2 = image.scale(img, rang1)
		local ret = mycrop(img2, rang2, fill_value)
		return ret
	end
end

function crop_imgs(data, rang)
	local rang = rang or 224
	local imsize = data:size()
	local dim = data:dim()

	if dim==3 then
		data = data[{{},{math.floor(imsize[2]/2)-rang/2+1,math.floor(imsize[2]/2)+rang/2},{math.floor(imsize[3]/2)-rang/2+1,math.floor(imsize[3]/2)+rang/2}}]
	elseif dim==4 then
		data = data[{{},{},{math.floor(imsize[3]/2)-rang/2+1,math.floor(imsize[3]/2)+rang/2},{math.floor(imsize[4]/2)-rang/2+1,math.floor(imsize[4]/2)+rang/2}}]
	end

	return data
end
