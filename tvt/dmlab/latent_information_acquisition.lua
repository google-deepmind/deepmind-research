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
local factory = require 'latent_information_acquisition_factory'

return factory.createLevelApi{
    episodeLengthSeconds = 40,
    exploreLengthSeconds = 5,
    distractorLengthSeconds = 30,
    numObjects = 3,
    probGoodObject = 0.5,
    correctReward = 20,
    incorrectReward = -10,
    differentDistractRoomTexture = true,
}
