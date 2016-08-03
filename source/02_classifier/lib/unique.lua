
------------------------------
-- function
------------------------------

-- Lua implements_func_table
-- http://lua.tips/index.php?Lua%20implements_func_table
function unique(tbl)
    local check = {}
    local res = {}
 
    for i, v in ipairs(tbl) do
        if not(check[v]) then
            check[v] = true
            res[1+#res] = v
        end
    end
 
    for k, v in pairs (tbl) do
        if not (type(k)=="number" and k%1==0) then
            res[k] = v
        end
    end
    return res
end
