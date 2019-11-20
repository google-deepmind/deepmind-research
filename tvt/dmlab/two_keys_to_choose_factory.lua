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
local make_map = require 'common.make_map'
local custom_observations = require 'decorators.custom_observations'
local debug_observations = require 'decorators.debug_observations'
local game = require 'dmlab.system.game'
local image_utils = require 'image_utils'
local map_maker = require 'dmlab.system.map_maker'
local maze_generation = require 'dmlab.system.maze_generation'
local pickup_decorator = require 'decorators.human_recognisable_pickups'
local random = require 'common.random'
local setting_overrides = require 'decorators.setting_overrides'
local texture_sets = require 'themes.texture_sets'
local themes = require 'themes.themes'
local hrp = require 'common.human_recognisable_pickups'

local EPISODE_LENGTH_SECONDS = 15
local EXPLORE_LENGTH_SECONDS = 5
local DISTRACTOR_LENGTH_SECONDS = 5
local CUE_COLORS = {2, 5}  -- Either red or blue cue.
local APPLE_ID = 998
local GOAL_ID = 999
local KEY_SPAWN_ID = 1000
local BAD_KEY_SPAWN_ID = 1001
local DOOR_ID = 1002
local KEY_CUE_RECTANGLE_WIDTH = 600
local KEY_CUE_RECTANGLE_HEIGHT = 200
local SHOW_COLOR_SQUARE_SECONDS = 1

-- Table that maps from full decal name to decal index number.
local decalIndices = {}

local EXPLORE_MAP = "exploreMap"
local DISTRACTOR_MAP = "distractorMap"
local REWARD_MAP = "rewardMap"
local COLORS = image_utils.COLORS

local GOAL_WITH_GOOD_KEY_REWARD = -1
local GOAL_WITH_BAD_KEY_REWARD = -10

local DISTRACTOR_ROOM_SIZE = {11, 11}
local EXPLORE_ROOM_SIZE = {4, 3}
local APPLE_REWARD = 5
local PROB_APPLE_IN_DISTRACTOR_MAP = 0.3
local DEFAULT_FINAL_REWARD = -20
local APPLE_EXTRA_REWARD_RANGE = 0
local DIFFERENT_DISTRACT_ROOM_TEXTURE = false


-- Set texture set for all maps.
local textureSet = texture_sets.TRON
local secondTextureSet = texture_sets.TETRIS

local REWARD_ROOM =[[
***
*P*
*H*
*G*
***
]]

local function createDistractorMaze(opts)
    -- Example room with height = 2, width = 3
    -- A are possible apple locations (everywhere)
    -- *****
    -- *APA*
    -- *AAA*
    -- *****

    local roomHeight = opts.roomSize[1]
    local roomWidth = opts.roomSize[2]
    centerWidth = 1 + math.ceil(roomWidth / 2)
    local maze = maze_generation:mazeGeneration{
        height = roomHeight + 2,  -- +2 for the two side of walls
        width = roomWidth + 2
    }

    -- Fill the room with 'A' for apples. updateSpawnVars decides which to use.
    for i = 2, roomHeight + 1 do
      for j = 2, roomWidth + 1 do
        maze:setEntityCell(i, j, 'A')
      end
    end
    -- Override one cell with 'P' for spawn point.
    maze:setEntityCell(2, centerWidth, 'P')

    print('Generated distractor maze with entity layer:')
    print(maze:entityLayer())
    io.flush()
    return maze
end

local function createExploreMaze(opts)
  -- Procedurelly generate room like below:
  -- xxxxxxx
  -- x  P  x
  -- x     x
  -- xK   Kx
  -- xxxxxxx

  local roomHeight = opts.roomSize[1]
  local roomWidth = opts.roomSize[2]
  centerWidth = 1 + math.ceil(roomWidth / 2)
  local maze = maze_generation:mazeGeneration{
      height = roomHeight + 2,
      width = roomWidth + 2
  }

  for i = 2, roomHeight + 1 do
    for j = 2, roomWidth + 1 do
      maze:setEntityCell(i, j, '.')
    end
  end

  maze:setEntityCell(2, centerWidth, 'P')
  maze:setEntityCell(roomHeight + 1, 2, 'K')
  maze:setEntityCell(roomHeight + 1, roomWidth + 1, 'K')

  print('Generated 2nd order explore maze with entity layer:')
  print(maze:entityLayer())
  io.flush()

  return maze
end

local factory = {}
game:console('cg_drawScriptRectanglesAlways 1')

function factory.createLevelApi(kwargs)
  kwargs.episodeLengthSeconds = kwargs.episodeLengthSeconds or
                                EPISODE_LENGTH_SECONDS
  kwargs.exploreLengthSeconds = kwargs.exploreLengthSeconds or
                                EXPLORE_LENGTH_SECONDS
  kwargs.distractorLengthSeconds = kwargs.distractorLengthSeconds or
                                   DISTRACTOR_LENGTH_SECONDS
  kwargs.distractorRoomSize = kwargs.distractorRoomSize or DISTRACTOR_ROOM_SIZE
  kwargs.probAppleInDistractorMap = kwargs.probAppleInDistractorMap or
                                    PROB_APPLE_IN_DISTRACTOR_MAP
  kwargs.exploreRoomSize = kwargs.exploreRoomSize or EXPLORE_ROOM_SIZE
  kwargs.appleExtraRewardRange =
      kwargs.appleExtraRewardRange or APPLE_EXTRA_REWARD_RANGE
  kwargs.differentDistractRoomTexture = kwargs.differentDistractRoomTexture or
                                        DIFFERENT_DISTRACT_ROOM_TEXTURE
  kwargs.defaultFinalReward = kwargs.defaultFinalReward or DEFAULT_FINAL_REWARD
  kwargs.goalWithGoodKeyReward = kwargs.goalWithGoodKeyReward or
                                 GOAL_WITH_GOOD_KEY_REWARD
  kwargs.goalWithBadKeyReward = kwargs.goalWithBadKeyReward or
                                GOAL_WITH_BAD_KEY_REWARD
  kwargs.appleReward = kwargs.appleReward or APPLE_REWARD

  local api = {}

  function api:init(params)
    self:_createSquareExploreMap()
    self:_createDistractorMap()
    self:_createRewardMap()


    -- key 1 is a red key, good, leads to less negative reward.
    local keyInfo = {shape='key', pattern='solid',
                     color1 = {255, 0, 0}, color2={0, 0, 0}}
    self._keyObject = hrp.create(keyInfo)
    self._keyCueRgba = {1, 0, 0, 1}

    -- key 2 is a blue key, bad, leads to more negative reward.
    local keyInfo2 = {shape='key', pattern='solid',
                     color1 = {0, 0, 255}, color2={0, 0, 0}}
    self._keyObject2 = hrp.create(keyInfo2)
    self._keyCueRgba2 = {0, 0, 1, 1}

    self._keyCueRgbaNoKey = {0, 0, 0, 1}
  end

  function api:_createRewardMap()

    self._rewardMap = map_maker:mapFromTextLevel{
        mapName = REWARD_MAP,
        entityLayer = REWARD_ROOM,
    }

    -- Create map theme and override default wall decal placement.
    local rewardMapTheme = themes.fromTextureSet{
        textureSet = textureSet,
        decalFrequency = 0.0,
    }

    self._rewardMap = map_maker:mapFromTextLevel{
        mapName = REWARD_MAP,
        entityLayer = REWARD_ROOM,
        theme = rewardMapTheme,
        callback = function (i, j, c, maker)
          local pickup = self:_makePickup(c)
          if pickup then
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:_createSquareExploreMap()
    -- Create a maze to be converted into map.
    local maze = createExploreMaze{
        roomSize = kwargs.exploreRoomSize
    }

    -- Create a map theme without wall decal placement.
    local exploreMapTheme = themes.fromTextureSet{
        textureSet = textureSet,
        decalFrequency = 0.0,
    }

    self._exploreMap = map_maker:mapFromTextLevel{
        mapName = EXPLORE_MAP,
        entityLayer = maze:entityLayer(),
        theme = exploreMapTheme,
        callback = function (i, j, c, maker)
          local pickup = self:_makePickup(c)
          if pickup then
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:_createDistractorMap()

    -- Create maze to be converted into map.
    local maze = createDistractorMaze{
        roomSize = kwargs.distractorRoomSize,
    }

    -- Create map theme with no wall decals.
    local texture = textureSet
    if kwargs.differentDistractRoomTexture then
      texture = secondTextureSet
    end
    local distractorMapTheme = themes.fromTextureSet{
        textureSet = texture,
        decalFrequency = 0.0,
    }

    self._exploreMap = map_maker:mapFromTextLevel{
        mapName = DISTRACTOR_MAP,
        entityLayer = maze:entityLayer(),
        theme = distractorMapTheme,
        callback = function (i, j, c, maker)
          local pickup = self:_makePickup(c)
          if pickup then
            return maker:makeEntity{i = i, j = j, classname = pickup}
          end
        end
    }
  end

  function api:start(episode, seed)
    random:seed(seed)

    self._map = nil
    self._time = 0
    self._holdingKey = false
    self._holdingBadKey = false
    self._keyPosCount = 0

    self._collectedGoal = false
    self._showKeyCue = false
    self._showNoKeyCue = false
    self._finalReward = kwargs.defaultFinalReward
    self._finalRewardAdded = false

    if kwargs.distractorLengthSecondsRange then
      self._distractorLen = random:uniformReal(
          kwargs.distractorLengthSecondsRange[1],
          kwargs.distractorLengthSecondsRange[2])
    else
      self._distractorLen = kwargs.distractorLengthSeconds
    end

    if kwargs.exploreRoomSize then
      local posIndex = {1, 2}  -- only 2 possible key location
      random:shuffleInPlace(posIndex)
      self._keyPosition = posIndex[1]
      self._keyPosition2 = posIndex[2]
    end

    -- Set instruction channel output to defaultFinalReward.
    -- Later this will be set to be the goal reward if collected.
    self.setInstruction(tostring(kwargs.defaultFinalReward))
  end

  function api:filledRectangles(args)
    if self._showKeyCue or self._showNoKeyCue then
      local cueColor
      if self._holdingKey then
        cueColor = self._keyCueRgba
      elseif self._holdingBadKey then
        cueColor = self._keyCueRgba2
      elseif self._showNoKeyCue then
        cueColor = self._keyCueRgbaNoKey
      end
      return {{
          x = 12,
          y = 12,
          width = KEY_CUE_RECTANGLE_WIDTH,
          height = KEY_CUE_RECTANGLE_HEIGHT,
          rgba = cueColor
      }}
    end
    return {}
  end

  function api:nextMap()
    -- 1. Decide what is the next map.
    if self._map == nil then
      self._map = EXPLORE_MAP
    elseif self._map == DISTRACTOR_MAP then
      self._map = REWARD_MAP
    elseif self._map == EXPLORE_MAP then
      if self._distractorLen > 0.0 then
        -- if not holding any key, show the no key cue
        if not self._holdingKey and not self._holdingBadKey then
          self._showNoKeyCue = true
          self._NoKeyCueTime = self._time
        end
        self._map = DISTRACTOR_MAP
      else
        self._map = REWARD_MAP
      end
    elseif self._map == REWARD_MAP then
      -- Stay in distractor map (no more apples) till the end of episode.
      self._map = DISTRACTOR_MAP
      self._collectedGoal = true
    end

    -- 2. Set up timeout for the up-coming map.
    if self._map == EXPLORE_MAP then
      self._timeOut = self._time + kwargs.exploreLengthSeconds
    elseif self._map == DISTRACTOR_MAP and not self._collectedGoal then
      self._timeOut = self._time + self._distractorLen
    elseif self._map == REWARD_MAP then
      self._timeOut = nil
    end

    return self._map
  end

 -- PICKUP functions ----------------------------------------------------------

  function api:_makePickup(c)
    if c == 'K' then
      return 'key'
    elseif c == 'G' then
      return 'goal'
    elseif c == 'A' then
      return 'apple_reward'
    end
  end

  function api:canPickup(spawnId)
    -- Cannot pick up another key if avatar is already holding a key.
    if spawnId == KEY_SPAWN_ID and self._holdingBadKey then
      return false
    end
    if spawnId == BAD_KEY_SPAWN_ID and self._holdingKey then
      return false
    end

    return true
  end

  function api:pickup(spawnId)
    if spawnId == GOAL_ID then
      local goalReward
      if self._holdingKey then
        goalReward = kwargs.goalWithGoodKeyReward
      elseif self._holdingBadKey then
        goalReward = kwargs.goalWithBadKeyReward
      end
      self.setInstruction(tostring(goalReward))
      game:addScore(-10)  -- offset the default +10 for pick up goal.
      self._finalReward = goalReward
      game:finishMap()
    end
    if spawnId == KEY_SPAWN_ID then
      self._holdingKey = true
      self._holdingKeyTime = self._time
      self._showKeyCue = true
    end
    if spawnId == BAD_KEY_SPAWN_ID then
      self._holdingBadKey = true
      self._holdingKeyTime = self._time
      self._showKeyCue = true
    end

    if spawnId == APPLE_ID then
      -- note the -1 for the default 1 point for apple in dmlab
      appleReward = kwargs.appleReward +
          random:uniformInt(0, kwargs.appleExtraRewardRange) - 1
      game:addScore(appleReward)
    end
  end

  function api:hasEpisodeFinished(timeSeconds)
    self._time = timeSeconds

    -- Give the final reward near the end of the episode.
    if not self._finalRewardAdded and
        timeSeconds > kwargs.episodeLengthSeconds - 0.1 then
        game:addScore(self._finalReward)
        self._finalRewardAdded = true
    end

    if (self._holdingKey or self._holdingBadKey) and
       self._time - self._holdingKeyTime > SHOW_COLOR_SQUARE_SECONDS then
      self._showKeyCue = false
    end

    if self._showNoKeyCue and
       self._time - self._NoKeyCueTime > SHOW_COLOR_SQUARE_SECONDS then
      self._showNoKeyCue = false
    end

    if self._map == EXPLORE_MAP or self._map == DISTRACTOR_MAP or
       self._map == SECOND_ORDER_EXPLORE_MAP then
      if timeSeconds > self._timeOut then
        game:finishMap()
      end
      return false
    end
  end

  function api:canTrigger(teleportId, targetName)
    if string.sub(targetName, 1, 4) == 'door' then
      -- open the door no matter which key the avatar holds.
      if self._holdingKey or self._holdingBadKey then
          return true
      else
          return false
      end
    end
    return false
  end

  function api:updateSpawnVars(spawnVars)
    local classname = spawnVars.classname
    if classname == "info_player_start" then
      -- Spawn facing South.
      spawnVars.angle = "-90"
      spawnVars.randomAngleRange = "0"
    elseif classname == "func_door" then
      spawnVars.id = tostring(DOOR_ID)
      spawnVars.wait = "1000000" --  Door open for a long time.
    elseif classname == "goal" then
      spawnVars.id = tostring(GOAL_ID)
    elseif classname == "apple_reward" then
      -- The avatar is spawned to distractor room after reaching goal
      -- there should be no more apples in such case.
      if self._collectedGoal then
        return nil
      end
      local useApple = false
      if kwargs.probAppleInDistractorMap > 0 then
        useApple = random:uniformReal(0, 1) < kwargs.probAppleInDistractorMap
        spawnVars.id = tostring(APPLE_ID)
      end
      if not useApple then
        return nil
      end
    elseif classname == "key" then
      self._keyPosCount = self._keyPosCount + 1
      if self._keyPosition == self._keyPosCount then
        spawnVars.id = tostring(KEY_SPAWN_ID)
        spawnVars.classname = self._keyObject
      elseif self._keyPosition2 == self._keyPosCount then
        spawnVars.id = tostring(BAD_KEY_SPAWN_ID)
        spawnVars.classname = self._keyObject2
      else
        return nil
      end
    end
    return spawnVars
  end

  custom_observations.decorate(api)
  pickup_decorator.decorate(api)
  setting_overrides.decorate{
      api = api,
      apiParams = kwargs,
      decorateWithTimeout = true
  }
  return api

end

return factory
