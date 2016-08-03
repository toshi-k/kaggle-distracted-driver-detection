
local Calibrator, parent = torch.class('nn.Calibrator','nn.Module')
 
function Calibrator:__init(n)
	parent.__init(self)
	self.n = n
	self.bias = torch.Tensor(n):fill(0)
	self.batch_bias = torch.Tensor(n):fill(0)
	self.gradBias = torch.Tensor(n):fill(0)
end

function Calibrator:updateOutput(input)

	self.output:resizeAs(input)
	self.output:copy(input)

	if input:dim() == 1 then
		self.output:add(self.bias)
	else
		self.batchSize = input:size(1)
		self.batch_bias:set(self.bias:repeatTensor(self.batchSize, 1))
		self.output:add(self.batch_bias)
	end
	return self.output
end

function Calibrator:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(gradOutput)
	self.gradInput:copy(gradOutput)
	return self.gradInput
end

-- avoid parameter update
function Calibrator:accGradParameters()
end
function Calibrator:accUpdateGradParameters()
end
function Calibrator:updateParameters()
end

-- update bias ( called in 6_calibrate.lua )
function Calibrator:updateBias(diff)
	self.bias:add(diff)
end
