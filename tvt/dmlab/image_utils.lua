-- Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
-- Licensed under the Apache License, Version 2.0 (the "License");
-- you may not use this file except in compliance with the License.
-- You may obtain a copy of the License at
--    http://www.apache.org/licenses/LICENSE-2.0
-- Unless required by applicable law or agreed to in writing, software
-- distributed under the License is distributed on an "AS IS" BASIS,
-- WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
-- See the License for the specific language governing permissions and
-- limitations under the License.
-- ============================================================================
local tensor = require 'dmlab.system.tensor'

local utils = {}
utils.COLORS = {
    {0, 0, 0},
    {0, 0, 170},
    {0, 170, 0},
    {0, 170, 170},
    {170, 0, 0},
    {170, 0, 170},
    {170, 85, 0},
    {170, 170, 170},
    {85, 85, 85},
    {85, 85, 255},
    {85, 255, 85},
    {85, 255, 255},
    {255, 85, 85},
    {255, 85, 255},
    {255, 255, 85},
    {255, 255, 255},
}

function utils:createByteImage(h, w, rgb)
  return tensor.ByteTensor(h, w, 4):fill{rgb[1], rgb[2], rgb[3], 255}
end

function utils:createTransparentImage(h, w)
  return tensor.ByteTensor(h, w, 4):fill{127, 127, 127, 0}
end

return utils
