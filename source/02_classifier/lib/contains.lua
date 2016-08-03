
------------------------------
-- function
------------------------------

-- http://lua-users.org/wiki/SetOperations
function contains(t, e)
	for i = 1,#t do
		if t[i] == e then return true end
	end
	return false
end
