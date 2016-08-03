
------------------------------
-- function
------------------------------

function mycrop(img, rang)
	local imsize = img:size()
	local x, y = 3, 2

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
	local ret = torch.Tensor(img:size(1), rang, rang):fill(0.5)
	ret[{{}, {rang/2+1-math.floor(imsize2[y]/2),rang/2+math.ceil(imsize2[y]/2)},{rang/2+1-math.floor(imsize2[x]/2),rang/2+math.ceil(imsize2[x]/2)}}] = crimg2

	return ret
end

function myresize(img, rang1, rang2)
	if (img:size(1) < rang2) and (img:size(2) < rang2) then
		local ret = mycrop(img, rang2)
		return ret
	else
		local img2 = image.scale(img, rang1)
		local ret = mycrop(img2, rang2)
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

function get_mergin(sum_vec)
	local p1 = 0
	for i=1,sum_vec:size()[1] do
		if sum_vec[i] > 1 then
			p1 = i
			break
		end
	end

	local p2 = sum_vec:size()[1]
	for i=sum_vec:size()[1],1,-1 do
		if sum_vec[i] > 1 then
			p2 = i
			break
		end
	end

	return p1, p2
end
